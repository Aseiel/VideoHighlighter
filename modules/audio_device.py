"""Keep the preview's audio on whatever the system currently calls default.

A ``QAudioOutput`` resolves ``QMediaDevices.defaultAudioOutput()`` exactly
once, when it is constructed, and then holds that device forever. Nothing in
Qt re-points it afterwards. So if the app is opened with speakers selected and
the user later switches Windows over to headphones, the preview keeps talking
to the speakers — or falls silent, if the old device disappeared entirely —
until the window is closed and reopened. Every other player on the machine
follows the switch, so the app looks broken rather than merely stubborn.

Register each output with :func:`follow_system_default` and one shared watcher
re-points them all when the default moves. The watcher listens to
``QMediaDevices.audioOutputsChanged`` and *also* re-checks on a slow timer:
the change notification is backend-specific, and a two-second poll comparing
one device id costs nothing next to the alternative of missing the switch.
"""

from PySide6.QtCore import QObject, QTimer
from PySide6.QtMultimedia import QMediaDevices

# How often to re-read the system default, as a backstop for the change
# notification. Slow enough to be free, fast enough to feel automatic.
_POLL_MS = 2000


class _DefaultDeviceWatcher(QObject):
    """Re-points every registered QAudioOutput when the default device moves."""

    def __init__(self):
        super().__init__()
        self._outputs = []
        self._devices = QMediaDevices(self)
        self._current_id = self._default_id()
        self._devices.audioOutputsChanged.connect(self._resync)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._resync)
        self._timer.start(_POLL_MS)

    @staticmethod
    def _default_id():
        device = QMediaDevices.defaultAudioOutput()
        return None if device.isNull() else bytes(device.id())

    def add(self, audio_output):
        if audio_output not in self._outputs:
            self._outputs.append(audio_output)

    def _resync(self):
        device = QMediaDevices.defaultAudioOutput()
        if device.isNull():
            return
        device_id = bytes(device.id())
        if device_id == self._current_id:
            return
        self._current_id = device_id

        # Deleted C++ objects raise on touch; drop them as we find them.
        alive = []
        for audio_output in self._outputs:
            try:
                audio_output.setDevice(device)
            except RuntimeError:
                continue
            alive.append(audio_output)
        self._outputs = alive
        print(f"[audio] default output is now {device.description()!r} "
              f"({len(alive)} player(s) re-pointed)")


_watcher = None


def follow_system_default(audio_output):
    """Make ``audio_output`` track the system default output device.

    Returns the same object, so it can wrap the construction inline. Never
    raises: an environment without a working multimedia backend just keeps
    Qt's original behaviour.
    """
    global _watcher
    try:
        if _watcher is None:
            _watcher = _DefaultDeviceWatcher()
        _watcher.add(audio_output)
    except Exception as e:
        print(f"[audio] cannot track default output device: {e}")
    return audio_output
