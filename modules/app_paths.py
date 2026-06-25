"""Path helpers that work both when running from source (``python main.py``)
and when bundled into a PyInstaller executable.

- Reading from source: paths resolve against the project root.
- Reading from an exe: bundled (read-only) resources live under ``sys._MEIPASS``;
  user-editable config lives next to the executable so edits persist.
"""

import os
import sys
import shutil


def _project_root() -> str:
    # modules/app_paths.py -> parent of the modules/ dir is the project root
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resource_path(filename: str) -> str:
    """Absolute path to a bundled, read-only resource (script or PyInstaller exe)."""
    if getattr(sys, "frozen", False):
        base = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    else:
        base = _project_root()
    return os.path.join(base, filename)


def user_data_dir() -> str:
    """Persistent, writable directory: next to the exe when frozen, else project root."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return _project_root()


def data_file(name: str) -> str:
    """Resolve a data/model file that may ship bundled but can be overridden by
    dropping a file of the same name next to the executable (or in the project
    root when run from source). The user copy wins; otherwise the bundled copy.

    This lets users swap in a retrained model on a packaged exe without rebuilding.
    From source both locations are the project root, so behaviour is unchanged.
    """
    user = os.path.join(user_data_dir(), name)
    if os.path.exists(user):
        return user
    return resource_path(name)


def ffmpeg_exe() -> str:
    """Resolve a usable ffmpeg executable.

    Order: system ffmpeg on PATH (what dev typically uses) -> the binary shipped
    with imageio-ffmpeg (bundled into the exe, so it works when the frozen app has
    no ffmpeg on PATH) -> bare "ffmpeg" as a last resort. Returns a path/name; the
    caller may still get FileNotFoundError if nothing is available.
    """
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.exists(exe):
            return exe
    except Exception:
        pass
    return "ffmpeg"


def config_path(filename: str = "config.yaml") -> str:
    """Resolve a user-editable config file.

    When frozen, this lives next to the executable (so edits/saves persist) and is
    seeded from the bundled default on first run. From source it's just the file in
    the project root, so ``python main.py`` behaves exactly as before.
    """
    target = os.path.join(user_data_dir(), filename)
    if not os.path.exists(target):
        bundled = resource_path(filename)
        try:
            if os.path.exists(bundled) and os.path.abspath(bundled) != os.path.abspath(target):
                shutil.copy2(bundled, target)
        except Exception:
            # Can't write next to the exe (e.g. read-only install) -> read the bundled copy
            return bundled
    return target
