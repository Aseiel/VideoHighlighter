"""'Teach a model to find this': the dialog half, shared by every video surface.

The Live overlay's right-click and the plain preview's (vr_video_view) both end
in the same two steps: ask what the thing is called, then, once a box is drawn,
seed a background-taught model from it (modules/teach/from_player.py). Each
surface draws the box its own way and hands over the box as fractions of the
whole frame.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import QInputDialog, QMessageBox

MENU_TEXT = "🧠  Teach a model to find this — draw a box…"
TITLE = "Teach a model"


def ask_model_name(parent) -> Optional[str]:
    """The name to teach under, or None if the user backed out."""
    name, ok = QInputDialog.getText(
        parent, TITLE,
        "What is it? (a name; the same name again adds another view)")
    name = name.strip() if ok else ""
    return name or None


def teach(parent, video_path: str, roi, seconds: float, name: str) -> None:
    """Seed a model from the box `roi` (x, y, w, h fractions of the whole frame)
    on the frame at `seconds`, and tell the user what happens next."""
    if not video_path:
        return
    from modules.teach.from_player import message, teach as seed

    try:
        result = seed(video_path, float(seconds), tuple(roi), name)
    except (ValueError, OSError) as exc:
        QMessageBox.warning(parent, TITLE, str(exc))
        return
    print(f"teach: seeded {result['class']!r} in {result['root']} "
          f"({result['seeds']} seeds)")
    QMessageBox.information(parent, TITLE, message(result))
