"""The macOS bundle: the simulator's contract, and the library it must keep.

Two unrelated-looking things that come from one afternoon. The 0.11.0 .app
could not analyse anything, because the build deleted a library OpenVINO links
against; and the way anyone here can look at macOS behaviour at all is a
simulator that fakes the environment, which is only safe if it puts everything
back.
"""

from __future__ import annotations

import os
import re
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKFLOW = os.path.join(_REPO, ".github", "workflows", "build-release.yaml")


class TestTheLibraryOpenVINONeeds:
    """`libtbb.12.dylib` was deleted from the bundle in March 2026 because it
    could not be ad-hoc signed. OpenVINO links against it, so `_pyopenvino`
    then failed to load, and `action_recognition` imports openvino at module
    scope — which took `pipeline` with it. Every macOS run died at import for
    six months.

    Deleting it is the tempting fix whenever signing complains again, so the
    prohibition is written down where it will be noticed.
    """

    def test_the_build_does_not_delete_it(self):
        text = open(WORKFLOW, encoding="utf-8").read()

        offenders = [line.strip() for line in text.splitlines()
                     if re.search(r"rm\s+.*libtbb\.12", line)
                     or re.search(r"rm\s+.*libtbb\.dylib", line) and "libtbb.12" in line]

        assert not offenders, (
            "the macOS build deletes the library OpenVINO loads: "
            f"{offenders}. Sign it instead — see the build step's comment.")

    def test_it_is_added_to_the_bundle(self):
        """Collected explicitly, because the loader looks for it at the root of
        Frameworks rather than inside the openvino package."""
        text = open(WORKFLOW, encoding="utf-8").read()

        assert "libtbb.12.dylib" in text, "nothing puts libtbb in the bundle"
        assert "--add-binary" in text

    def test_the_numa_pieces_are_still_dropped(self):
        """tbbbind links Homebrew's libhwloc by absolute path — a path no user
        has — and TBB runs without it."""
        text = open(WORKFLOW, encoding="utf-8").read()

        assert re.search(r"rm\s+-f\s+\"\$OV_LIBS\"/libtbbbind\*", text)
        assert re.search(r"rm\s+-f\s+\"\$OV_LIBS\"/libhwloc\*", text)


simulate = pytest.importorskip("tools.simulate_macos_bundle")


class TestTheSimulatorPutsEverythingBack:
    """It patches `sys.platform`, `sys.frozen`, `sys.executable`, `HOME` and the
    working directory. Any one of those surviving the block would make every
    test that ran afterwards lie."""

    def test_the_process_is_unchanged_afterwards(self, tmp_path):
        before = (sys.platform, getattr(sys, "frozen", None), sys.executable,
                  os.getcwd(), os.environ.get("HOME"))

        with simulate.frozen_macos_app(str(tmp_path)):
            assert sys.platform == "darwin"
            assert getattr(sys, "frozen", False) is True

        after = (sys.platform, getattr(sys, "frozen", None), sys.executable,
                 os.getcwd(), os.environ.get("HOME"))
        assert before == after

    def test_it_restores_even_when_the_body_raises(self, tmp_path):
        before = (sys.platform, os.getcwd())

        with pytest.raises(RuntimeError):
            with simulate.frozen_macos_app(str(tmp_path)):
                raise RuntimeError("the run blew up")

        assert (sys.platform, os.getcwd()) == before

    def test_it_writes_nothing_outside_its_own_directory(self, tmp_path):
        """An earlier draft chdir'ed to the real drive root and left a `cache`
        folder there. A simulation does not get to litter the machine."""
        with simulate.frozen_macos_app(str(tmp_path)):
            cwd = os.getcwd()

        assert os.path.commonpath([cwd, str(tmp_path)]) == str(tmp_path)
