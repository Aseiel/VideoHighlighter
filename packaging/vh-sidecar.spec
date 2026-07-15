# PyInstaller spec for the FastAPI sidecar bundled into the Tauri app.
#
# The sidecar imports the same engine as main.py (pipeline -> torch/cv2/whisper/
# ultralytics/openvino), so the collection flags mirror .github/workflows/
# build-release.yaml. Differences from the Qt build:
#   * console app, not --windowed: it's a child process, never user-facing, and
#     its stdout is piped to the Tauri log.
#   * no PySide6/Qt: the sidecar never draws anything.
#   * onedir, not onefile: a onefile build unpacks ~2GB of torch/openvino to a
#     temp dir on every launch, which would add many seconds to app startup.
#
# Build:  pyinstaller packaging/vh-sidecar.spec --noconfirm
# Output: dist/vh-sidecar/vh-sidecar.exe

import os

from PyInstaller.utils.hooks import (
    collect_all,
    collect_data_files,
    collect_submodules,
    copy_metadata,
)

# Relative paths in a spec resolve against the spec's own directory, not the
# invocation cwd — anchor everything to the project root explicitly.
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(SPEC)), ".."))


def _p(*parts):
    return os.path.join(ROOT, *parts)


datas = [
    (_p("config.yaml"), "."),
    (_p("yolo_objects_labels.json"), "."),
    (_p("kinetics_400_labels.json"), "."),
    (_p("modules"), "modules"),
]

# Optional data dirs — only present in some checkouts.
for d in ("models", "assets"):
    if os.path.isdir(_p(d)):
        datas.append((_p(d), d))

binaries = []
hiddenimports = [
    # The engine's top-level modules are imported lazily inside worker.py, so
    # PyInstaller's static analysis can't see them.
    "action_recognition",
    "crop_actions",
    "downloader",
    "object_recognition",
    "pipeline",
    "sorter",
    "llm.clip_prefilter",
    "llm.llm_module",
    "sidecar.worker",
    # uvicorn resolves these by string at runtime.
    "uvicorn.logging",
    "uvicorn.loops.auto",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan.on",
]

hiddenimports += collect_submodules("whisper")
hiddenimports += collect_submodules("ultralytics")
hiddenimports += collect_submodules("optimum")
hiddenimports += collect_submodules("transformers")

datas += collect_data_files("whisper")
datas += collect_data_files("ultralytics")
datas += collect_data_files("transformers")

# transformers/optimum read their own versions via importlib.metadata at import.
for pkg in ("transformers", "tokenizers", "huggingface-hub", "safetensors",
            "regex", "optimum", "optimum-intel"):
    try:
        datas += copy_metadata(pkg)
    except Exception:
        pass

ov_datas, ov_binaries, ov_hidden = collect_all("openvino")
datas += ov_datas
binaries += ov_binaries
hiddenimports += ov_hidden

# imageio_ffmpeg ships the ffmpeg binary the pipeline falls back to when none is
# on PATH (see app_paths.ffmpeg_exe).
from PyInstaller.utils.hooks import collect_dynamic_libs

binaries += collect_dynamic_libs("imageio_ffmpeg")
datas += collect_data_files("imageio_ffmpeg")


a = Analysis(
    [_p("sidecar", "server.py")],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[_p("packaging", "pyinstaller-hooks")],
    hooksconfig={},
    runtime_hooks=[],
    # nncf is an optional openvino extra that drags in a lot and isn't used.
    excludes=["nncf", "PySide6", "PyQt5", "matplotlib.backends._backend_tk",
              "tkinter"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="vh-sidecar",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="vh-sidecar",
)
