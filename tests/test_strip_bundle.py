"""tools/strip_bundle.py on both bundle layouts.

The mac build never ran the strip step: Qt WebEngine (Chromium) and OpenVINO's
readers for formats the app never opens shipped in every .dmg. A macOS .app
also cross-links its two halves, so a removed folder can leave links behind,
which codesign then refuses.
"""
from __future__ import annotations

import os

import pytest

from tools import strip_bundle


def _write(path, size=1024):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(b"\0" * size)


def _symlink_or_skip(target, link):
    try:
        os.symlink(target, link, target_is_directory=os.path.isdir(target))
    except (OSError, NotImplementedError):
        pytest.skip("this machine cannot make symlinks")


def test_windows_layout_keeps_its_own_patterns(tmp_path):
    root = tmp_path / "VideoHighlighter"
    _write(str(root / "_internal/PySide6/Qt6WebEngineCore.dll"), 4096)
    _write(str(root / "_internal/PySide6/Qt6Core.dll"))
    assert strip_bundle.strip(str(root)) == 4096
    assert (root / "_internal/PySide6/Qt6Core.dll").exists()
    assert not (root / "_internal/PySide6/Qt6WebEngineCore.dll").exists()


def test_a_mac_app_loses_webengine_and_unused_openvino_readers(tmp_path):
    app = tmp_path / "VideoHighlighter.app"
    fw = app / "Contents/Frameworks"
    _write(str(fw / "PySide6/Qt/lib/QtWebEngineCore.framework/Versions/A/QtWebEngineCore"), 8192)
    _write(str(fw / "PySide6/Qt/lib/QtCore.framework/Versions/A/QtCore"))
    _write(str(fw / "PySide6/QtWebEngineCore.abi3.so"))
    _write(str(fw / "openvino/libs/libopenvino_tensorflow_frontend.2026.4.0.dylib"), 2048)
    _write(str(fw / "openvino/libs/libopenvino_onnx_frontend.2026.4.0.dylib"))
    _write(str(fw / "openvino/libs/libopenvino_pytorch_frontend.2026.4.0.dylib"))

    removed = strip_bundle.strip(str(app))

    assert removed == 8192 + 1024 + 2048
    assert (fw / "PySide6/Qt/lib/QtCore.framework/Versions/A/QtCore").exists()
    assert (fw / "openvino/libs/libopenvino_onnx_frontend.2026.4.0.dylib").exists()
    assert (fw / "openvino/libs/libopenvino_pytorch_frontend.2026.4.0.dylib").exists()
    assert not (fw / "PySide6/Qt/lib/QtWebEngineCore.framework").exists()


def test_links_into_a_removed_folder_go_with_it(tmp_path):
    app = tmp_path / "VideoHighlighter.app"
    real = app / "Contents/Frameworks/PySide6/Qt/lib/QtWebEngineCore.framework"
    _write(str(real / "QtWebEngineCore"))
    res = app / "Contents/Resources/PySide6/Qt/lib"
    os.makedirs(res)
    _symlink_or_skip(str(real), str(res / "QtWebEngineCore.framework"))
    # A link to something that stays must survive.
    keep = app / "Contents/Frameworks/PySide6/Qt/lib/QtCore.framework"
    _write(str(keep / "QtCore"))
    _symlink_or_skip(str(keep), str(res / "QtCore.framework"))

    strip_bundle.strip(str(app))

    assert not os.path.lexists(res / "QtWebEngineCore.framework")
    assert os.path.exists(res / "QtCore.framework")


def test_report_groups_by_path_and_skips_links(tmp_path, capsys):
    root = tmp_path / "VideoHighlighter"
    _write(str(root / "_internal/torch/lib/libtorch_cpu.dll"), 3 * 2**20)
    _write(str(root / "_internal/cv2/cv2.pyd"), 2**20)
    strip_bundle.report(str(root), depth=2)
    out = capsys.readouterr().out
    assert "bundle size: 4 MB in 2 files" in out
    assert out.index("_internal/torch") < out.index("_internal/cv2")
