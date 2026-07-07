"""Full-frame video surface.

A plain full-frame video widget used as the timeline preview. Side-by-side
(VR/3D) footage is shown as-is (both eyes); there is no left-eye crop in this
build. The ``set_vr_mode`` / ``set_native_resolution`` hooks are kept as no-ops
so the shared timeline viewer code runs unchanged and VR toggles are harmless.
"""

from __future__ import annotations

from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.QtMultimediaWidgets import QVideoWidget


class VRVideoView(QWidget):
    """Plain full-frame video surface used as the timeline preview."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background-color: black;")
        self.setMinimumSize(320, 240)
        self._player = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._video = QVideoWidget(self)
        layout.addWidget(self._video)

    @property
    def video_output(self):
        """The video surface — pass to ``player.setVideoOutput(...)``."""
        return self._video

    def attach_player(self, player) -> None:
        """Remember the player so the surface can track it."""
        self._player = player

    def set_vr_mode(self, enabled: bool) -> None:
        """No-op: the left-eye (side-by-side) crop is not available in this build."""
        return None

    def set_native_resolution(self, w: int, h: int) -> None:
        """No-op — the full frame is shown at its natural aspect ratio."""
        return None
