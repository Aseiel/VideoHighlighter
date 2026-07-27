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
        self._quick = None
        self._video_widget = None
        self._fallback = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if not self._init_qml(layout):
            self._init_fallback(layout)

    def _init_qml(self, layout) -> bool:
        """Try the GPU QML VideoOutput (needed for the VR left-eye crop).

        Returns False — so the caller falls back to a plain QVideoWidget — when
        QML isn't usable. In a packaged build that happens if QtQuick or the
        VRVideoOutput.qml data file wasn't bundled; loading here rather than
        crashing means the app still shows video (just without the VR crop)."""
        try:
            from PySide6.QtQuickWidgets import QQuickWidget
            quick = QQuickWidget()
            quick.setResizeMode(QQuickWidget.ResizeMode.SizeRootObjectToView)
            try:
                quick.setClearColor(Qt.GlobalColor.black)
            except Exception:
                pass
            qml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "VRVideoOutput.qml")
            quick.setSource(QUrl.fromLocalFile(qml_path))
            for err in quick.errors():
                print(f"⚠️ VRVideoView QML error: {err.toString()}")
            if quick.status() == QQuickWidget.Status.Error or quick.rootObject() is None:
                print("⚠️ VRVideoView: QML surface failed to load; falling back to "
                      "a plain QVideoWidget (VR left-eye crop disabled).")
                quick.deleteLater()
                return False
            self._quick = quick
            layout.addWidget(quick)
            return True
        except Exception as e:  # noqa: BLE001 — any QtQuick/QML problem -> fall back
            print(f"⚠️ VRVideoView: QML unavailable ({e}); falling back to a plain "
                  f"QVideoWidget (VR left-eye crop disabled).")
            return False

    def _init_fallback(self, layout) -> None:
        """Plain QVideoWidget — needs no QML runtime, so it works in any build.
        Loses the GPU left-eye VR crop, but always shows the video."""
        from PySide6.QtMultimediaWidgets import QVideoWidget
        self._fallback = True
        self._video_widget = QVideoWidget()
        layout.addWidget(self._video_widget)

    @property
    def video_output(self):
        """The QML VideoOutput item — pass to ``player.setVideoOutput(...)``.

        Fetched via ``findChild`` (typed as plain ``QObject``) because PySide6
        can't marshal the concrete ``QQuickVideoOutput*`` through ``property()``.
        """
        if self._fallback:
            return self._video_widget
        root = self._quick.rootObject() if self._quick is not None else None
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
        its OSD exactly once regardless of how often VR is toggled.

        No-op in the QVideoWidget fallback (that path can't GPU-crop)."""
        if self._fallback or self._quick is None:
            return
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
        if self._fallback or self._quick is None:
            return
        root = self._quick.rootObject()
        if root is not None and w > 0 and h > 0:
            root.setProperty("nativeWidth", float(w))
            root.setProperty("nativeHeight", float(h))
