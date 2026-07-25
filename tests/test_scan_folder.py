"""
Tests for sidecar.server._scan_video_files — the batch picker's folder scan.

Pins the four behaviours the endpoint promises: only video extensions are
returned (case-insensitive), results are natural-sorted (clip2 before clip10),
recursion is opt-in (off by default the tree below the top level is invisible),
and the 2000-file cap holds.

Pure filesystem logic — no server, no ffmpeg — so it runs in the shimmed CI env.
"""

from __future__ import annotations

import os

from sidecar.server import _SCAN_CAP, _scan_video_files


def _touch(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("x")


def _build_tree(root):
    # Top level: a mix of video and non-video files, deliberately out of order.
    _touch(os.path.join(root, "clip10.mp4"))
    _touch(os.path.join(root, "clip2.MP4"))       # uppercase ext
    _touch(os.path.join(root, "clip1.mov"))
    _touch(os.path.join(root, "notes.txt"))       # not a video
    _touch(os.path.join(root, "poster.jpg"))      # not a video
    # Nested level: only reached when recursive.
    _touch(os.path.join(root, "sub", "deep.mkv"))
    _touch(os.path.join(root, "sub", "deep.txt"))


def test_extension_filter_and_natural_sort(tmp_path):
    _build_tree(str(tmp_path))
    files = _scan_video_files(str(tmp_path), recursive=False)
    names = [os.path.basename(f) for f in files]
    # Non-video files excluded; natural sort puts clip2 before clip10.
    assert names == ["clip1.mov", "clip2.MP4", "clip10.mp4"]


def test_recursive_off_ignores_subtree(tmp_path):
    _build_tree(str(tmp_path))
    files = _scan_video_files(str(tmp_path), recursive=False)
    assert all("sub" not in os.path.relpath(f, str(tmp_path)).split(os.sep)[:-1]
               for f in files)
    assert not any(os.path.basename(f) == "deep.mkv" for f in files)


def test_recursive_on_walks_subtree(tmp_path):
    _build_tree(str(tmp_path))
    files = _scan_video_files(str(tmp_path), recursive=True)
    names = {os.path.basename(f) for f in files}
    assert "deep.mkv" in names
    assert "deep.txt" not in names
    # 3 top-level videos + 1 nested video.
    assert len(files) == 4


def test_case_insensitive_extensions(tmp_path):
    for name in ("a.WEBM", "b.MoV", "c.M2TS", "d.avI"):
        _touch(os.path.join(str(tmp_path), name))
    files = _scan_video_files(str(tmp_path), recursive=False)
    assert len(files) == 4


def test_cap_is_enforced(tmp_path):
    # One over the cap; the helper must stop at _SCAN_CAP.
    for i in range(_SCAN_CAP + 5):
        _touch(os.path.join(str(tmp_path), f"v{i:05d}.mp4"))
    files = _scan_video_files(str(tmp_path), recursive=False)
    assert len(files) == _SCAN_CAP
