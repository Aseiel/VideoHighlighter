"""Path helpers that work both when running from source (``python main.py``)
and when bundled into a PyInstaller executable.

- Reading from source: paths resolve against the project root.
- Reading from an exe: bundled (read-only) resources live under ``sys._MEIPASS``;
  user-editable config lives next to the executable so edits persist.
"""

import functools
import os
import sys
import shutil


def _project_root() -> str:
    # modules/system/app_paths.py -> up three, the parent of modules/, is
    # the project root.
    return os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )


@functools.lru_cache(maxsize=8)
def _is_writable(path: str) -> bool:
    """Can this process create a file in ``path``?

    Asked rather than assumed, and asked by *trying*: on Windows,
    ``os.access(path, os.W_OK)`` answers from the read-only attribute and says
    yes for a directory whose ACL will refuse the write. Cached, because this
    is consulted on every path lookup in the app.
    """
    probe = os.path.join(path, f".vh-write-test-{os.getpid()}")
    try:
        with open(probe, "w"):
            pass
    except OSError:
        return False
    try:
        os.remove(probe)
    except OSError:
        pass
    return True


def _per_user_dir() -> str:
    """Where this platform keeps an application's own files, created if needed."""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser(
            r"~\AppData\Local")
    elif sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Application Support")
    else:
        base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser(
            "~/.local/share")
    target = os.path.join(base, "VideoHighlighter")
    os.makedirs(target, exist_ok=True)
    return target


def resource_path(filename: str) -> str:
    """Absolute path to a bundled, read-only resource (script or PyInstaller exe)."""
    if getattr(sys, "frozen", False):
        base = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    else:
        base = _project_root()
    return os.path.join(base, filename)


def user_data_dir() -> str:
    """Persistent, writable directory for everything the app keeps: the analysis
    cache, the debug log, stats, custom models, composition rules.

    Beside the executable when that folder can be written to, which keeps a
    portable install self-contained — copy the folder, keep your caches. When it
    cannot, the platform's own per-user location is used instead.

    **That fallback is not a nicety.** Where the install folder refuses writes,
    every one of those fails, and the app then only works when it is started as
    an administrator — something a user discovers by accident and then has to
    remember forever. Elevation is not required by anything this app does; it
    was only ever a way of making the writes land somewhere.

    **macOS never writes beside the executable.** There ``sys.executable`` lives
    inside ``VideoHighlighter.app/Contents/MacOS``, and a bundle is read-only
    under Gatekeeper's translocation, breaks its own signature when written to,
    and need not be writable by the user at all.
    """
    if not getattr(sys, "frozen", False):
        return _project_root()

    if sys.platform == "darwin":
        try:
            return _per_user_dir()
        except OSError:
            # A home we cannot write to is not worth crashing over here; the
            # caller's own failure will say more than a guess would.
            return os.path.dirname(sys.executable)

    beside_exe = os.path.dirname(sys.executable)
    if _is_writable(beside_exe):
        return beside_exe
    try:
        return _per_user_dir()
    except OSError:
        return beside_exe


def use_writable_cwd() -> str:
    """Move the process into :func:`user_data_dir` when frozen; returns the cwd.

    Dozens of call sites default to a relative path — ``./cache`` most of all —
    and a relative path is only as good as the directory the process happens to
    be in. macOS launches an .app with the working directory set to ``/``, which
    is read-only, so every one of them failed with ``[Errno 30] Read-only file
    system: 'cache'`` the moment a video was opened.

    Windows launches an exe from its own folder, so this is a no-op there in
    practice; doing it explicitly keeps it true however the app was started — a
    shortcut with its own "Start in", a file association, a terminal somewhere
    else.

    Bundled resources are unaffected: they resolve through
    :func:`resource_path`, which is absolute.
    """
    if not getattr(sys, "frozen", False):
        return os.getcwd()
    target = user_data_dir()
    try:
        os.makedirs(target, exist_ok=True)
        os.chdir(target)
    except OSError as e:
        print(f"⚠️ could not work from {target}: {e}")
    return os.getcwd()


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


def action_models_dir() -> str:
    """Managed folder for trained action-recognition models.

    Sits beside ``models/custom/`` (object detectors) so everything a training
    run produces lands under ``models/`` rather than in the install root.
    """
    return os.path.join(user_data_dir(), "models", "actions")


def action_model_file(name: str) -> str:
    """Resolve one of the fixed action-model slots for *reading*.

    ``models/actions/<name>`` is where training writes and where the app looks
    first. The flat locations ``data_file()`` checks stay as a fallback: models
    trained before this folder existed live in the root, and a packaged exe
    still honours a file dropped next to it. When neither exists the managed
    path is returned, so a caller reporting "not found" names the new place.
    """
    managed = os.path.join(action_models_dir(), name)
    if os.path.exists(managed):
        return managed
    legacy = data_file(name)
    if os.path.exists(legacy):
        return legacy
    return managed


def latest_custom_pose_model():
    """Custom keypoint models are not supported: the only trainer for them was
    AGPL, and nothing trained with it may ship. Kept so callers need no guard."""
    return None


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


def object_model_names(path: str) -> list:
    """Class names a detector reports: its embedded metadata, else the
    labels.json beside it. [] when neither is readable."""
    from modules.vision.detection_backend import names_from_model, load_class_names
    try:
        return names_from_model(path) or load_class_names(
            os.path.join(os.path.dirname(path), "labels.json"))
    except Exception as e:
        print(f"⚠️ could not read classes from {os.path.basename(path)}: {e}")
        return []


def import_object_model(src: str) -> str:
    """Install a detector into models/custom/<name>/ and return the model path.

    Each model gets its own folder because a YOLOX export names its classes in
    a ``labels.json`` beside it, and two models sharing one folder would share
    one labels file. An .xml brings its .bin; a labels.json next to the source
    comes along.
    """
    name = os.path.splitext(os.path.basename(src))[0]
    dst_dir = os.path.join(object_models_dir(), name)
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, os.path.basename(src))
    shutil.copy2(src, dst)
    src_dir = os.path.dirname(src)
    if src.lower().endswith(".xml"):
        bin_src = os.path.splitext(src)[0] + ".bin"
        if os.path.exists(bin_src):
            shutil.copy2(bin_src, os.path.join(dst_dir, os.path.basename(bin_src)))
    labels_src = os.path.join(src_dir, "labels.json")
    if os.path.exists(labels_src):
        shutil.copy2(labels_src, os.path.join(dst_dir, "labels.json"))
    return dst


def discover_object_models() -> list:
    """List custom object detectors under models/custom/, newest first, each with
    the class names it reports.

    Returns [{"path": str, "name": str, "classes": list[str]}]. Empty when the
    folder is absent or holds no models — callers then offer only the standard
    option. Only .onnx and OpenVINO .xml are listed: those are what the YOLOX
    runtime loads.
    """
    import glob
    d = object_models_dir()
    if not os.path.isdir(d):
        return []
    paths = []
    for ext in ("*.onnx", "*.xml"):
        paths.extend(glob.glob(os.path.join(d, ext)))
        paths.extend(glob.glob(os.path.join(d, "*", ext)))
    # An installed model sits as .onnx + .xml side by side; list it once, as IR.
    paths = [p for p in paths if not (
        p.lower().endswith(".onnx") and os.path.exists(os.path.splitext(p)[0] + ".xml"))]
    paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)

    # lazy: avoid import cycle
    from modules.vision.detection_backend import names_from_model, load_class_names
    out = []
    for p in paths:
        # Same resolution order build_object_detector uses — embedded metadata
        # first, then a labels.json sidecar. A YOLOX export carries no
        # class-name metadata, so without the sidecar it would list zero classes.
        names = names_from_model(p) or load_class_names(
            os.path.join(os.path.dirname(p), "labels.json"))
        if not names:
            continue
        out.append({
            "path": p,
            "name": os.path.splitext(os.path.basename(p))[0],
            "classes": names,
        })
    # Community models installed from the hub (model_hub), listed after the
    # user's own. Each is checksum-verified here; a changed file drops out.
    try:
        from model_hub.hub import installed_detectors
        out.extend(installed_detectors())
    except Exception as e:  # noqa: BLE001 - a broken install must not hide the rest
        print(f"⚠️ community model discovery failed: {e}")
    return out


def custom_action_decoder_paths() -> tuple[str, str, str]:
    """Where to read the user's custom fine-tuned OpenVINO action decoder from:
    (xml, bin, labels_json).

    ``models/actions/`` first — that is where the Intel trainer writes and where
    the Advanced tab's "Import model…" button installs — then the legacy flat
    locations ``data_file()`` resolves, so an older install keeps working."""
    return (
        action_model_file("action_classifier_3d.xml"),
        action_model_file("action_classifier_3d.bin"),
        action_model_file("intel_finetuned_classifier_3d_mapping.json"),
    )


def import_custom_action_model(decoder_xml_src: str, labels_json_src: str = "") -> int:
    """Install a user-trained OpenVINO action decoder (+ its .bin, + a labels
    mapping) into the writable user-data location, so it's picked up in place
    of the bundled default — mirrors object_models_dir()'s import flow, but
    for the single custom-action-decoder slot (no multi-model discovery here).

    ``labels_json_src`` may be omitted — a same-named ``*.json`` next to
    ``decoder_xml_src`` is used automatically if present (what the training
    pipeline writes alongside the decoder).

    Returns the number of classes found in the installed labels file (0 if
    none), so the caller can report/validate the import.
    """
    # Always install into the managed folder, never over a legacy root copy the
    # reader may still be resolving: models/actions/ wins the lookup, so the
    # freshly imported model is the one that loads.
    dest = action_models_dir()
    os.makedirs(dest, exist_ok=True)
    dst_xml = os.path.join(dest, "action_classifier_3d.xml")
    dst_bin = os.path.join(dest, "action_classifier_3d.bin")
    dst_labels = os.path.join(dest, "intel_finetuned_classifier_3d_mapping.json")
    shutil.copy2(decoder_xml_src, dst_xml)

    src_bin = os.path.splitext(decoder_xml_src)[0] + ".bin"
    if os.path.exists(src_bin):
        shutil.copy2(src_bin, dst_bin)

    if not labels_json_src:
        candidate = os.path.splitext(decoder_xml_src)[0] + ".json"
        if os.path.exists(candidate):
            labels_json_src = candidate

    n_classes = 0
    if labels_json_src and os.path.exists(labels_json_src):
        shutil.copy2(labels_json_src, dst_labels)
        try:
            import json
            with open(dst_labels, "r", encoding="utf-8") as f:
                n_classes = len(json.load(f).get("idx_to_label", {}))
        except Exception:
            pass
    return n_classes


def r3d_custom_action_paths() -> tuple[str, str]:
    """Where to read the user's custom fine-tuned R3D (PyTorch) action model
    from: (weights_pth, mapping_json). ``models/actions/`` first, then the
    legacy flat locations — same rule as custom_action_decoder_paths()."""
    return (
        action_model_file("r3d_finetuned.pth"),
        action_model_file("r3d_finetuned_mapping.json"),
    )


def import_r3d_action_model(weights_pth_src: str, mapping_json_src: str = "") -> tuple[int, str]:
    """Install a user-trained R3D (PyTorch) action model (+ its mapping JSON)
    into the writable user-data location, alongside import_custom_action_model()
    but for the R3D slot.

    Unlike the OpenVINO decoder, the mapping JSON is effectively required — it
    carries both the class labels (``idx_to_label``) and the
    ``metadata.model_variant`` (r3d_18 / mc3_18 / r2plus1d_18) the loader needs
    to rebuild the right architecture before loading the weights. A same-named
    ``*.json`` next to the ``.pth`` is used automatically if present.

    Returns ``(num_classes, model_variant)`` — ``model_variant`` is ``""`` when
    the mapping omits it, in which case the loader falls back to the UI's
    "R3D model variant" dropdown selection.
    """
    dest = action_models_dir()
    os.makedirs(dest, exist_ok=True)
    dst_pth = os.path.join(dest, "r3d_finetuned.pth")
    dst_mapping = os.path.join(dest, "r3d_finetuned_mapping.json")
    shutil.copy2(weights_pth_src, dst_pth)

    if not mapping_json_src:
        candidate = os.path.splitext(weights_pth_src)[0] + ".json"
        if os.path.exists(candidate):
            mapping_json_src = candidate

    n_classes, variant = 0, ""
    if mapping_json_src and os.path.exists(mapping_json_src):
        shutil.copy2(mapping_json_src, dst_mapping)
        try:
            import json
            with open(dst_mapping, "r", encoding="utf-8") as f:
                data = json.load(f)
            n_classes = len(data.get("idx_to_label", {}))
            variant = (data.get("metadata") or {}).get("model_variant", "") or ""
        except Exception:
            pass
    return n_classes, variant


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
