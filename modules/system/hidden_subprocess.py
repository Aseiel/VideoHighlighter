"""No console window for the programs the app starts, on Windows.

The packaged app is --windowed: it has no console. When a process without one
starts a console program (ffmpeg, ffprobe, yt-dlp, 7-Zip), Windows opens a new
console window for it, and every cut, probe and download flashed one up. Only
a handful of the ~50 places that start such a program asked for
``CREATE_NO_WINDOW``, and Whisper's own audio loader, which runs ffmpeg, is not
ours to change. A launcher in between makes it worse: Chocolatey's
``C:\\ProgramData\\chocolatey\\bin\\ffmpeg.exe`` is a small console program
that starts the real ffmpeg, so each run flashed two windows (measured: 10 for
5 runs from a windowless Python; 0 with the flag).

So :func:`install` makes ``CREATE_NO_WINDOW`` the default for every
``subprocess.Popen`` in this process, once, at startup. A caller that asks for
a console of its own (``CREATE_NEW_CONSOLE``) or none at all
(``DETACHED_PROCESS``) keeps exactly what it asked for; a GUI program ignores
the flag, so opening Explorer or relaunching the app is unchanged.

Only when the app has no console: run from a terminal, the children share it,
and their output belongs there.
"""

from __future__ import annotations

import functools
import subprocess
import sys

CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
CREATE_NEW_CONSOLE = getattr(subprocess, "CREATE_NEW_CONSOLE", 0x00000010)
DETACHED_PROCESS = getattr(subprocess, "DETACHED_PROCESS", 0x00000008)

_installed = False


def _has_console() -> bool:
    try:
        import ctypes
        return bool(ctypes.windll.kernel32.GetConsoleWindow())
    except Exception:  # noqa: BLE001 - assume one, and change nothing
        return True


def install(force: bool = False) -> bool:
    """Hide the console windows of child processes. True when it took effect.

    A no-op off Windows, when the process has a console (unless ``force``),
    and on a second call.
    """
    global _installed
    if _installed or sys.platform != "win32" or (not force and _has_console()):
        return False
    original = subprocess.Popen.__init__

    @functools.wraps(original)
    def __init__(self, *args, **kwargs):
        flags = kwargs.get("creationflags") or 0
        if not flags & (CREATE_NEW_CONSOLE | DETACHED_PROCESS):
            kwargs["creationflags"] = flags | CREATE_NO_WINDOW
        original(self, *args, **kwargs)

    subprocess.Popen.__init__ = __init__
    _installed = True
    print("hidden_subprocess: child processes start without a console window")
    return True
