"""The list of local models the report can be written with, and how it is stored.

One model was enough while the narrator did one job. It does two now — advising
on the settings and reading the footage — and they suit different models: a small
instruction-tuned one answers "which weight should I raise" in a sentence, where
reading a scene wants something larger, and a vision model wants a projector file
alongside it that the single-model setting had nowhere to put.

So the app keeps a list and the user picks per run. Everything here is the pure
half of that: parsing what was stored, migrating what was stored before, and
naming an entry for a menu. The Qt half lives with the rest of the window.

Stored as JSON in one settings value rather than as a settings group per model,
because the alternative is inventing keys at runtime and then discovering which
of them still exist — and because a list the user can copy between machines in
one line is worth more than a tidier registry they cannot.
"""
from __future__ import annotations

import json
import os
from typing import Mapping, Optional, Sequence

# The backends `llm.llm_module` knows how to build. A stored entry naming
# anything else is dropped on load: it cannot be constructed, and offering it in
# a menu only moves the failure to the point where the user has waited for it.
BACKENDS = ("ollama", "llama-cpp")


def _clean(entry: Mapping) -> Optional[dict]:
    """One stored entry, or ``None`` when it cannot be used."""
    if not isinstance(entry, Mapping):
        return None
    backend = str(entry.get("backend") or "").strip()
    model = str(entry.get("model") or "").strip()
    if backend not in BACKENDS or not model:
        return None
    out = {"backend": backend, "model": model}
    mmproj = str(entry.get("mmproj") or "").strip()
    if mmproj:
        out["mmproj"] = mmproj
    label = str(entry.get("label") or "").strip()
    if label:
        out["label"] = label
    return out


def parse(raw) -> list:
    """The stored list, with anything unusable dropped.

    Accepts the already-decoded list as well as the JSON text, because QSettings
    on Windows hands back whichever it feels like depending on how the value was
    written and by which version of Qt.
    """
    if not raw:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return []
    if isinstance(raw, Mapping):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        return []
    out = []
    for entry in raw:
        cleaned = _clean(entry)
        if cleaned and cleaned not in out:
            out.append(cleaned)
    return out


def serialise(models: Sequence[Mapping]) -> str:
    """The list as one settings value."""
    return json.dumps([m for m in (_clean(x) for x in models or ()) if m])


def label_for(model: Optional[Mapping]) -> str:
    """What to call an entry in a menu.

    A GGUF path is the thing users actually configure and the thing that will
    not fit: the file name identifies it and the directory almost never does, so
    the name is what shows and the full path goes in the tooltip.
    """
    if not model:
        return "no model"
    label = str(model.get("label") or "").strip()
    if label:
        return label
    name = str(model.get("model") or "")
    if model.get("backend") == "llama-cpp":
        name = os.path.basename(name.replace("\\", "/")) or name
        if name.lower().endswith(".gguf"):
            name = name[:-5]
    return name or "unnamed"


def migrate(models: Sequence[Mapping],
            backend: Optional[str],
            model: Optional[str]) -> list:
    """Fold the older single-model setting into the list, once.

    Someone upgrading has a model configured and no list. Losing it would make
    the first run after an update fail in the one place they have no reason to
    look, so it becomes the first entry instead.
    """
    models = list(models or ())
    existing = _clean({"backend": backend, "model": model})
    if existing and existing not in models:
        models.insert(0, existing)
    return models


def active(models: Sequence[Mapping], chosen: Optional[str]) -> Optional[dict]:
    """The entry the user last picked, or the first one as a default."""
    models = list(models or ())
    if not models:
        return None
    for entry in models:
        if label_for(entry) == chosen:
            return entry
    return models[0]
