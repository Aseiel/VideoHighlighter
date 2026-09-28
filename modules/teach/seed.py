"""Start teaching an object from one box: "this one".

The easiest thing to ask of anyone is to point. A box drawn around the thing
on one frame of a video (in the player, or given by an agent that can see
the frame) is:

* the class, created if it is new;
* its first example: the crop is what "looks like it" means from now on,
  far closer than the class's name would be (``boxes.crop_vectors``);
* the video it came from, added as footage to search (``find``).

Seed again, on other frames or videos, to show it from other sides; each
seed is an accepted box and sharpens the search.
"""
from __future__ import annotations

import os

from modules.teach.boxes import SEED_SOURCE, store
from modules.teach.project import OBJECTS, Project
from modules.vision.label_store import ACCEPTED, LabelledBox


def parse_box(text) -> tuple:
    """``"x,y,w,h"`` (fractions of the frame) -> a checked tuple."""
    values = [float(v) for v in (text.split(",") if isinstance(text, str) else text)]
    if len(values) != 4:
        raise ValueError("a box is x,y,w,h as fractions of the frame, e.g. 0.4,0.3,0.2,0.25")
    x, y, w, h = values
    if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > 1.0001 or y + h > 1.0001:
        raise ValueError(f"box {text!r} is not inside the frame (x,y,w,h as fractions)")
    return (x, y, w, h)


def seed(root: str, video: str, moment: float, box, name: str,
         description: str = "") -> dict:
    """Record one drawn box; create the project and the class if needed."""
    if not os.path.isfile(video):
        raise FileNotFoundError(video)
    box = parse_box(box)
    if os.path.exists(os.path.join(root, "project.json")):
        project = Project.load(root)
    else:
        project = Project.create(root, OBJECTS)
    if project.task != OBJECTS:
        raise ValueError("a box shows an object: seed an object project")
    created = project.get_class(name) is None
    if created:
        project.add_class(name, description)
    source = project.add_source(video)
    project.save()
    labels = store(project)
    labels.add(LabelledBox(video=source.path, time=float(moment), class_name=name,
                           box=box, source=SEED_SOURCE, confidence=1.0,
                           verdict=ACCEPTED))
    labels.save()
    seeds = sum(1 for b in labels.accepted()
                if b.class_name == name and b.source == SEED_SOURCE)
    return {"class": name, "created": created, "seeds": seeds, "source": source.id}
