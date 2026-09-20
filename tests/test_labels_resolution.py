"""A labels file named without a directory belongs to the app, not to the cwd.

The packaged build works from its user-data directory (``app_paths.use_writable_cwd``)
while its data ships under ``_MEIPASS``, so every ``load_class_names("yolo_objects_labels.json")``
call looked beside the exe, found nothing, and returned no names. The detector
was then dropped for want of labels -- with its *model* found, because
``DEFAULT_MODEL_DIR`` is absolute -- and a 0.12.0 run reported:

    COCO labels not found: yolo_objects_labels.json
    Object detection unavailable -- no usable model

From source the cwd *is* the project root, which is why nothing showed there and
why this is a cwd test: it is the only difference between the two.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from modules.vision.detection_backend import load_class_names

REPO = Path(__file__).resolve().parent.parent
COCO = "yolo_objects_labels.json"


def test_bare_name_resolves_from_an_unrelated_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert not (tmp_path / COCO).exists()
    names = load_class_names(COCO)
    assert names[:1] == ["person"]
    assert len(names) == 80


def test_a_file_in_the_cwd_still_wins(tmp_path, monkeypatch):
    """Unchanged behaviour from source, and the override an editable copy is for."""
    (tmp_path / COCO).write_text(json.dumps(["widget"]), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert load_class_names(COCO) == ["widget"]


def test_a_path_with_a_directory_is_left_alone(tmp_path, monkeypatch):
    """Only a bare name means "the app's own"; a real path stays a real path."""
    monkeypatch.chdir(REPO)
    missing = tmp_path / "models" / COCO
    assert load_class_names(str(missing)) == []


def test_the_labels_ship_where_the_resolver_looks():
    """The fix relies on app_paths finding the bundled copy; assert it is there."""
    from modules.system import app_paths
    assert os.path.exists(app_paths.data_file(COCO))
