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


def latest_custom_pose_model():
    """Return the most recent trained custom keypoint model (best.pt), or None.

    Looks in the usual ultralytics output locations under the project, plus a
    drop-in copy next to the executable / project root named
    'custom_keypoints.pt'. Lets the GUI/pipeline pick up a freshly trained model
    without hardcoding a path.
    """
    import glob
    roots = {_project_root(), user_data_dir()}
    candidates = []
    for root in roots:
        # explicit drop-in
        for name in ("custom_keypoints.pt",):
            p = os.path.join(root, name)
            if os.path.exists(p):
                candidates.append(p)
        # ultralytics training outputs
        candidates += glob.glob(os.path.join(root, "**", "weights", "best.pt"), recursive=True)
        candidates += glob.glob(os.path.join(root, "training", "**", "weights", "best.pt"), recursive=True)
    candidates = [c for c in candidates if os.path.exists(c)]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def _read_keypoint_names(path):
    import json
    try:
        if path and os.path.exists(path):
            data = json.load(open(path, encoding="utf-8"))
            names = data if isinstance(data, list) else data.get("keypoint_names")
            return [str(x) for x in (names or []) if str(x).strip()]
    except Exception:
        pass
    return []


def custom_keypoint_names():
    """The custom model's keypoint names (its detectable 'classes').
    Resolution order: sidecar next to the model -> labeler_keypoints.json ->
    any exported label JSON's keypoint_names.
    """
    import glob
    model = latest_custom_pose_model()
    if model:
        names = _read_keypoint_names(os.path.join(os.path.dirname(model), "keypoint_names.json"))
        if names:
            return names
    for root in {_project_root(), user_data_dir()}:
        names = _read_keypoint_names(os.path.join(root, "labeler_keypoints.json"))
        if names:
            return names
        for f in glob.glob(os.path.join(root, "labels", "*.json")):
            names = _read_keypoint_names(f)
            if names:
                return names
    return []


def object_models_dir() -> str:
    """Managed folder for custom object detectors, auto-discovered in the
    Advanced tab. Sits next to the executable when frozen so imported models
    survive a restart; the project root when running from source."""
    return os.path.join(user_data_dir(), "models", "custom")


_OBJECT_MODEL_NAMES_CACHE: dict = {}


def object_model_names(path: str) -> list:
    """Class names a detector reports for itself, or [] if it is not an object
    detector or cannot be read.

    Cached per (path, mtime): this loads the model, which is slow enough that
    re-reading it every time the combo rebuilds is noticeable.
    """
    try:
        key = (path, os.path.getmtime(path))
    except OSError:
        return []
    if key in _OBJECT_MODEL_NAMES_CACHE:
        return _OBJECT_MODEL_NAMES_CACHE[key]
    names = []
    try:
        from ultralytics import YOLO
        model = YOLO(str(path))
        if getattr(model, "task", "") == "detect":
            names = [str(n) for n in model.names.values()]
    except Exception as e:
        print(f"⚠️ could not read classes from {os.path.basename(path)}: {e}")
    _OBJECT_MODEL_NAMES_CACHE[key] = names
    return names


def discover_object_models() -> list:
    """List custom object detectors under models/custom/, newest first, each with
    the class names read from the model itself.

    Returns [{"path": str, "name": str, "classes": list[str]}]. Empty when the
    folder is absent or holds no detectors — callers then offer only the standard
    option. Keypoint/pose models are skipped: they run a different code path and
    get their names from custom_keypoint_names().
    """
    import glob
    d = object_models_dir()
    if not os.path.isdir(d):
        return []
    paths = []
    for ext in ("*.pt", "*.onnx"):
        paths.extend(glob.glob(os.path.join(d, ext)))
    paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    out = []
    for p in paths:
        classes = object_model_names(p)
        if not classes:
            continue
        out.append({
            "path": p,
            "name": os.path.splitext(os.path.basename(p))[0],
            "classes": classes,
        })
    return out


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


def composition_rules_path() -> str | None:
    """Path to composition_rules.yaml (private, gitignored).
    Returns None when the file does not exist (engine is skipped)."""
    for candidate in (
        os.path.join(user_data_dir(), "composition_rules.yaml"),
        resource_path("composition_rules.yaml"),
    ):
        if os.path.exists(candidate):
            return candidate
    return None


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
