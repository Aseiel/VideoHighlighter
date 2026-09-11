"""Where a packaged app is allowed to write, and where it works from.

Both halves of one bug. A macOS .app launches with the working directory set to
``/`` and lives inside a bundle that must be treated as read-only, so the dozens
of call sites that default to ``./cache`` all failed with "[Errno 30] Read-only
file system: 'cache'" the moment a video was opened.

Everything here runs on any platform: the frozen macOS case is `sys.frozen` and
`sys.platform` monkeypatched, because that is exactly what the code reads.
"""

from __future__ import annotations

import os
import sys

import pytest

from modules import app_paths


@pytest.fixture
def frozen_mac(monkeypatch, tmp_path):
    """A frozen build on macOS, with a home directory we can inspect."""
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(sys, "executable",
                        str(tmp_path / "VideoHighlighter.app" / "Contents" /
                            "MacOS" / "VideoHighlighter"))
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))   # expanduser on Windows
    return home


@pytest.fixture
def frozen_windows(monkeypatch, tmp_path):
    exe_dir = tmp_path / "VideoHighlighter"
    exe_dir.mkdir()
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "executable", str(exe_dir / "VideoHighlighter.exe"))
    return exe_dir


class TestUserDataDir:
    def test_macos_writes_to_application_support(self, frozen_mac):
        """Not inside the .app: that is read-only under Gatekeeper's
        translocation, writing there breaks the signature, and an app in
        /Applications need not be writable by the user at all."""
        # Compared with forward slashes: the path is a macOS one whatever
        # platform the test itself runs on.
        target = app_paths.user_data_dir().replace(os.sep, "/")

        assert target.endswith("Library/Application Support/VideoHighlighter")
        assert ".app" not in target
        assert os.path.isdir(target), "the directory must exist, not just resolve"

    def test_windows_writes_beside_the_exe_when_it_can(self, frozen_windows):
        """A portable install stays self-contained: copy the folder, keep the
        caches. Moving that unconditionally would strand every existing user's
        cache and config."""
        assert app_paths.user_data_dir() == str(frozen_windows)

    def test_an_unwritable_install_falls_back_instead_of_needing_admin(
            self, frozen_windows, monkeypatch, tmp_path):
        """Where the install folder refuses writes, every write beside the exe
        fails, and the app then only works when started as an administrator —
        something the user discovers by accident and then has to remember.
        Nothing here needs elevation; it was only ever a way of making the
        writes land somewhere."""
        local = tmp_path / "AppData" / "Local"
        local.mkdir(parents=True)
        monkeypatch.setenv("LOCALAPPDATA", str(local))
        monkeypatch.setattr(app_paths, "_is_writable", lambda path: False)

        target = app_paths.user_data_dir()

        assert target == str(local / "VideoHighlighter")
        assert os.path.isdir(target)

    def test_writability_is_decided_by_trying(self, tmp_path):
        """os.access answers from the read-only attribute on Windows and says
        yes for a directory whose ACL will refuse the write, so the probe
        actually creates a file."""
        app_paths._is_writable.cache_clear()

        assert app_paths._is_writable(str(tmp_path)) is True
        assert app_paths._is_writable(str(tmp_path / "does-not-exist")) is False
        assert not list(tmp_path.iterdir()), "the probe left something behind"

    def test_from_source_it_is_the_project_root(self, monkeypatch):
        monkeypatch.setattr(sys, "frozen", False, raising=False)

        assert os.path.isfile(os.path.join(app_paths.user_data_dir(), "main.py"))

    def test_an_unwritable_home_falls_back_rather_than_raising(
            self, frozen_mac, monkeypatch):
        def no(*_args, **_kwargs):
            raise OSError("read-only")

        monkeypatch.setattr(app_paths.os, "makedirs", no)

        assert app_paths.user_data_dir() == os.path.dirname(sys.executable)


class TestWorkingDirectory:
    def test_a_frozen_app_moves_somewhere_writable(self, frozen_mac, monkeypatch):
        """The actual fix for `./cache` on macOS: `/` is read-only, so the
        process does not stay there."""
        seen = []
        monkeypatch.setattr(app_paths.os, "chdir", lambda p: seen.append(p))
        monkeypatch.setattr(app_paths.os, "getcwd", lambda: seen[-1] if seen else "/")

        result = app_paths.use_writable_cwd()

        assert seen and seen[0].endswith("VideoHighlighter")
        assert "Application Support" in seen[0]
        assert result == seen[0]

    def test_running_from_source_changes_nothing(self, monkeypatch):
        """A developer's shell is their own; `python main.py` must not move it."""
        monkeypatch.setattr(sys, "frozen", False, raising=False)
        monkeypatch.setattr(app_paths.os, "chdir",
                            lambda p: pytest.fail(f"chdir to {p} from source"))

        assert app_paths.use_writable_cwd() == os.getcwd()

    def test_a_failed_chdir_is_reported_not_raised(self, frozen_windows,
                                                   monkeypatch, capsys):
        """Whatever goes wrong here, the app still has to start."""
        def no(_path):
            raise OSError("nope")

        monkeypatch.setattr(app_paths.os, "chdir", no)

        app_paths.use_writable_cwd()

        assert "could not work from" in capsys.readouterr().out
