"""Teach a model from the player: draw a box, give it a name, done.

The one thing asked of the person is to point at the thing once. That box
becomes a seed (``seed``) in a project named after the thing, under the
app's user data, with ``background`` on: from then on, while the app is
idle, the project cuts the video, finds the thing in it, accepts what it is
sure of, trains, and installs the model only if it beats the last one
(``background``). The few boxes it is unsure of wait under Train -> From
videos -> Check guesses. Drawing another box with the same name, on another
frame or video, shows it from another side.
"""
from __future__ import annotations

from modules.teach.cli import resolve_root
from modules.teach.project import Project
from modules.teach.seed import seed


def teach(video: str, moment: float, roi, name: str) -> dict:
    """Seed ``name`` from a normalised ``roi`` at ``moment`` of ``video``."""
    name = " ".join(str(name).split())
    if not name:
        raise ValueError("give it a name")
    root = resolve_root(name)
    result = seed(root, video, moment, tuple(roi), name)
    project = Project.load(root)
    if not project.settings.background:
        project.settings.background = True
        project.save()
    return {**result, "root": root}


def message(result: dict) -> str:
    """What the person is told after drawing the box."""
    if result["seeds"] > 1:
        return (f"Another view of “{result['class']}” added ({result['seeds']} so far). "
                "Each one widens what it finds.")
    return (f"Learning to find “{result['class']}”. While the app is idle it looks "
            "for it in this video, accepts what it is sure of and trains a model, "
            "which is used only once it does better. It keeps a few questions for you "
            "under Train → From videos → Check guesses. Draw another box with the "
            "same name to show it from another side.")
