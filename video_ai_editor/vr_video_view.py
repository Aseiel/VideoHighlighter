"""Left-eye half-frame view for side-by-side (VR/3D) video.

Shows only the LEFT eye of a side-by-side frame, cropped **on the GPU** via a
QML ``VideoOutput`` inside a clipped QML ``Item`` (see ``VRVideoOutput.qml``).

Earlier attempts each failed on one axis:

* ``QVideoSink`` -> ``toImage()`` -> cropped ``QLabel`` — pure software, no
  swapchain (RTSS couldn't attach), couldn't hold the source framerate.
* ``QGraphicsVideoItem`` on an OpenGL ``QGraphicsView`` — crop worked, but each
  4K frame is copied to a ``QImage`` and repainted on the CPU (~80% CPU even
  paused, ~8-10 fps).
* Native ``QVideoWidget`` stretched + window-clipped — full framerate, but Qt6
  renders video into a composited RHI surface that *ignores window clipping*, so
  it showed a zoomed/overflowing frame instead of a crop.

QML ``clip`` is a scene-graph (GPU) clip that actually clips the hardware video,
so this keeps playback GPU-accelerated at the true source framerate **and**
crops correctly. Wire the player to the QML ``VideoOutput`` item with
``player.setVideoOutput(view.video_output)`` (Qt6 extracts its sink), and call
``view.attach_player(player)`` so the crop tracks the real resolution.
"""

from __future__ import annotations

import os

from PySide6.QtCore import Qt, QObject, QSize, QUrl
from PySide6.QtWidgets import QWidget, QVBoxLayout


class VRVideoView(QWidget):
    """QML VideoOutput cropped to the left eye of a side-by-side frame."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background-color: black;")
        self.setMinimumSize(320, 240)
        self._player = None

        from PySide6.QtQuickWidgets import QQuickWidget

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._quick = QQuickWidget()
        self._quick.setResizeMode(QQuickWidget.ResizeMode.SizeRootObjectToView)
        try:
            self._quick.setClearColor(Qt.GlobalColor.black)
        except Exception:
            pass
        qml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "VRVideoOutput.qml")
        self._quick.setSource(QUrl.fromLocalFile(qml_path))
        for err in self._quick.errors():
            print(f"⚠️ VRVideoView QML error: {err.toString()}")
        layout.addWidget(self._quick)

    @property
    def video_output(self):
        """The QML VideoOutput item — pass to ``player.setVideoOutput(...)``.

        Fetched via ``findChild`` (typed as plain ``QObject``) because PySide6
        can't marshal the concrete ``QQuickVideoOutput*`` through ``property()``.
        """
        root = self._quick.rootObject()
        return root.findChild(QObject, "vrVideoOut") if root is not None else None

    def attach_player(self, player) -> None:
        """Track the player's video resolution so the crop stays undistorted."""
        self._player = player
        try:
            player.metaDataChanged.connect(self._refresh_resolution)
        except Exception:
            pass
        self._refresh_resolution()

    def set_vr_mode(self, enabled: bool) -> None:
        """Toggle the SBS left-eye crop on the QML surface.

        ``True``  -> crop to the left eye (side-by-side VR/3D video).
        ``False`` -> show the whole frame (normal playback).

        This flips a property on the *same* live surface — no widget or swapchain
        is created or destroyed — which is what keeps RTSS/MSI-Afterburner drawing
        its OSD exactly once regardless of how often VR is toggled."""
        root = self._quick.rootObject()
        if root is not None:
            root.setProperty("vrMode", bool(enabled))

    def _refresh_resolution(self) -> None:
        if self._player is None:
            return
        try:
            from PySide6.QtMultimedia import QMediaMetaData
            res = self._player.metaData().value(QMediaMetaData.Key.Resolution)
        except Exception:
            res = None
        if isinstance(res, QSize) and res.width() > 0 and res.height() > 0:
            self.set_native_resolution(res.width(), res.height())

    def set_native_resolution(self, w: int, h: int) -> None:
        root = self._quick.rootObject()
        if root is not None and w > 0 and h > 0:
            root.setProperty("nativeWidth", float(w))
            root.setProperty("nativeHeight", float(h))
