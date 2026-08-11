"""Tests for downloading and applying an update.

No network and no real install: the HTTP opener is injected and everything
happens in tmp dirs. What's pinned down is that bad bytes never reach the
install, and that a failure leaves the install exactly as it was — the two
properties that decide whether a self-updater is safe to ship.
"""
from __future__ import annotations

import hashlib
import io
import os

import pytest

from modules import update_apply, update_download
from modules.update_manifest import UpdatePlan


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _entry(path, data):
    return {"path": path, "size": len(data), "sha256": _sha(data)}


def _plan(*entries):
    plan = UpdatePlan(version="0.9.1")
    plan.download = list(entries)
    return plan


BASE = "https://host/0.9.1"


def _opener(files: dict, fail: set = frozenset()):
    """Serve ``{relative_path: bytes}``; raise for anything in ``fail``."""
    class _Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.close()
            return False

    # Content-addressed: the host serves blobs at /files/<sha256>, so the fake
    # host is keyed the same way the real one would be.
    by_digest = {_sha(data): data for data in files.values()}
    fail_digests = {_sha(files[name]) for name in fail if name in files}

    def opener(url, headers):
        assert url.startswith(BASE + "/files/"), f"unexpected url: {url}"
        digest = url[len(BASE) + len("/files/"):]
        if digest in fail_digests:
            raise OSError("connection reset")
        if digest not in by_digest:
            raise OSError(f"404 {digest}")
        return _Response(by_digest[digest])

    return opener


def _write(root, relative, data: bytes):
    path = os.path.join(str(root), relative.replace("/", os.sep))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(data)
    return path


# --- downloading -----------------------------------------------------------

def test_downloads_and_verifies(tmp_path):
    data = b"new build bytes"
    plan = _plan(_entry("_internal/app.py", data))
    result = update_download.download_plan(
        plan, "https://host/0.9.1", str(tmp_path / "stage"),
        opener=_opener({"_internal/app.py": data}))

    assert result.ok
    assert result.staged == ["_internal/app.py"]
    staged = tmp_path / "stage" / "_internal" / "app.py"
    assert staged.read_bytes() == data


def test_corrupted_download_is_rejected_and_not_staged(tmp_path):
    """A host serving the wrong bytes must not produce a staged file.

    Content addressing does not make this impossible — a compromised or broken
    host can still answer /files/<sha> with something else. The hash check on
    what actually arrived is what catches it.
    """
    class _Wrong(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.close()
            return False

    def lying_host(url, headers):
        return _Wrong(b"something else entirely")

    plan = _plan(_entry("app.py", b"the real thing"))
    result = update_download.download_plan(
        plan, BASE, str(tmp_path / "stage"), opener=lying_host)

    assert not result.ok
    assert result.failed[0][1] == "hash mismatch"
    assert not (tmp_path / "stage" / "app.py").exists()
    assert not (tmp_path / "stage" / "app.py.part").exists()


def test_network_error_is_reported_not_raised(tmp_path):
    plan = _plan(_entry("a.py", b"x"), _entry("b.py", b"y"))
    result = update_download.download_plan(
        plan, "https://host/0.9.1", str(tmp_path / "stage"),
        opener=_opener({"a.py": b"x", "b.py": b"y"}, fail={"b.py"}))

    assert result.staged == ["a.py"]
    assert result.failed[0][0] == "b.py"
    assert not result.ok


def test_already_staged_files_are_not_refetched(tmp_path):
    data = b"already here"
    _write(tmp_path / "stage", "a.py", data)
    plan = _plan(_entry("a.py", data))

    def explode(url, headers):
        raise AssertionError("should not download an already-staged file")

    result = update_download.download_plan(
        plan, "https://host/0.9.1", str(tmp_path / "stage"), opener=explode)
    assert result.ok and result.staged == ["a.py"]


def test_stale_staged_file_is_replaced(tmp_path):
    _write(tmp_path / "stage", "a.py", b"old attempt")
    data = b"correct bytes"
    plan = _plan(_entry("a.py", data))
    result = update_download.download_plan(
        plan, "https://host/0.9.1", str(tmp_path / "stage"),
        opener=_opener({"a.py": data}))
    assert result.ok
    assert (tmp_path / "stage" / "a.py").read_bytes() == data


def test_cancel_stops_and_leaves_install_alone(tmp_path):
    plan = _plan(_entry("a.py", b"x"), _entry("b.py", b"y"))
    result = update_download.download_plan(
        plan, "https://host/0.9.1", str(tmp_path / "stage"),
        opener=_opener({"a.py": b"x", "b.py": b"y"}),
        should_cancel=lambda: True)
    assert result.cancelled
    assert not result.ok


def test_progress_is_reported(tmp_path):
    data = b"z" * 1000
    plan = _plan(_entry("a.py", data))
    seen = []
    update_download.download_plan(
        plan, "https://host/0.9.1", str(tmp_path / "stage"),
        opener=_opener({"a.py": data}),
        progress=lambda done, total, path: seen.append((done, total, path)))
    assert seen and seen[-1][0] == 1000 and seen[-1][1] == 1000


def test_unsafe_path_is_never_written(tmp_path):
    plan = _plan({"path": "../escape.py", "size": 1, "sha256": _sha(b"x")})
    result = update_download.download_plan(
        plan, "https://host/0.9.1", str(tmp_path / "stage"),
        opener=_opener({"../escape.py": b"x"}))
    assert result.failed == [("../escape.py", "unsafe path")]
    assert not (tmp_path / "escape.py").exists()


# --- applying --------------------------------------------------------------

def test_apply_replaces_files_and_keeps_the_old_one_aside(tmp_path):
    root, stage = tmp_path / "app", tmp_path / "stage"
    _write(root, "app.exe", b"v1")
    _write(stage, "app.exe", b"v2")

    result = update_apply.apply_update(str(root), str(stage), ["app.exe"])

    assert result.ok
    assert (root / "app.exe").read_bytes() == b"v2"
    # The displaced copy is still around until the next launch sweeps it.
    assert (root / update_apply.TRASH_DIRNAME / "app.exe").read_bytes() == b"v1"


def test_apply_adds_new_files(tmp_path):
    root, stage = tmp_path / "app", tmp_path / "stage"
    root.mkdir()
    _write(stage, "_internal/new.py", b"new")
    result = update_apply.apply_update(str(root), str(stage), ["_internal/new.py"])
    assert result.ok
    assert (root / "_internal" / "new.py").read_bytes() == b"new"


def test_apply_removes_deleted_files(tmp_path):
    root, stage = tmp_path / "app", tmp_path / "stage"
    _write(root, "gone.py", b"old")
    stage.mkdir()
    result = update_apply.apply_update(str(root), str(stage), [], ["gone.py"])
    assert result.ok
    assert not (root / "gone.py").exists()


def test_apply_writes_the_new_manifest_last(tmp_path):
    root, stage = tmp_path / "app", tmp_path / "stage"
    _write(root, "app.exe", b"v1")
    _write(stage, "app.exe", b"v2")
    update_apply.apply_update(str(root), str(stage), ["app.exe"],
                              manifest_bytes=b'{"version": "0.9.1"}')
    assert (root / "manifest.json").read_bytes() == b'{"version": "0.9.1"}'


def test_a_failed_apply_rolls_everything_back(tmp_path):
    """The property that matters most: no half-updated install."""
    root, stage = tmp_path / "app", tmp_path / "stage"
    _write(root, "a.py", b"a-v1")
    _write(root, "b.py", b"b-v1")
    _write(stage, "a.py", b"a-v2")
    # b.py is listed as staged but was never downloaded -> apply fails partway.

    result = update_apply.apply_update(str(root), str(stage), ["a.py", "b.py"])

    assert not result.ok
    assert "staged file missing" in result.error
    assert (root / "a.py").read_bytes() == b"a-v1", "first file must be restored"
    assert (root / "b.py").read_bytes() == b"b-v1"
    assert result.replaced == []


def test_apply_refuses_unsafe_paths(tmp_path):
    root, stage = tmp_path / "app", tmp_path / "stage"
    root.mkdir()
    _write(stage, "x", b"x")
    result = update_apply.apply_update(str(root), str(stage), ["../escape.py"])
    assert not result.ok
    assert not (tmp_path / "escape.py").exists()


def test_sweep_removes_the_old_files(tmp_path):
    root = tmp_path / "app"
    _write(root, f"{update_apply.TRASH_DIRNAME}/app.exe", b"old bytes")
    freed = update_apply.sweep_old(str(root))
    assert freed == len(b"old bytes")
    assert not (root / update_apply.TRASH_DIRNAME).exists()


def test_sweep_is_a_no_op_when_there_is_nothing(tmp_path):
    assert update_apply.sweep_old(str(tmp_path)) == 0


# --- the whole round trip --------------------------------------------------

def test_download_then_apply_end_to_end(tmp_path):
    root, stage = tmp_path / "app", tmp_path / "stage"
    _write(root, "app.exe", b"v1")
    _write(root, "_internal/keep.bin", b"unchanged")

    new_exe = b"v2 with the fix"
    plan = _plan(_entry("app.exe", new_exe))

    downloaded = update_download.download_plan(
        plan, "https://host/0.9.1", str(stage),
        opener=_opener({"app.exe": new_exe}))
    assert downloaded.ok

    applied = update_apply.apply_update(
        str(root), str(stage), downloaded.staged,
        manifest_bytes=b'{"version":"0.9.1"}')
    assert applied.ok

    assert (root / "app.exe").read_bytes() == new_exe
    assert (root / "_internal" / "keep.bin").read_bytes() == b"unchanged", \
        "untouched files must not be disturbed"

    update_apply.sweep_old(str(root))
    assert not (root / update_apply.TRASH_DIRNAME).exists()
