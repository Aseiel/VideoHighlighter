"""
Keep every shipped version string tied to `version.py`.

Why this exists
===============
The version lives in four places that no single build step rewrites:

  * `version.py`                          - read at runtime by the Qt app
                                            (`main.py`) and the FastAPI sidecar
                                            (`sidecar/server.py`, /health).
  * `frontend/package.json`               - the npm package version.
  * `frontend/src-tauri/tauri.conf.json`  - stamped into the installer and the
                                            Windows file properties.
  * `frontend/src-tauri/Cargo.toml`       - the Rust crate version (mirrored in
                                            Cargo.lock).

They drifted once already: `version.py` said 0.8.1 while the Tauri shell shipped
as 0.1.0 and npm as 0.0.0, so an installer built from this tree advertised a
version the running app disagreed with. The release workflow now derives the tag
from `version.py`, which makes the other three silently wrong rather than
merely inconsistent. This test is the guard.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent


def _expected_version() -> str:
    """Read __version__ out of version.py without importing it.

    version.py is trivial, but importing it drags in nothing and returns
    nothing useful for the *file* contract we care about, so parse the literal.
    """
    text = (REPO_ROOT / "version.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', text, re.M)
    assert match, "version.py has no parseable __version__ assignment"
    return match.group(1)


def test_version_py_is_a_plain_semver():
    """The release workflow uses this string as a git tag and an archive name."""
    version = _expected_version()
    assert re.fullmatch(r"\d+\.\d+\.\d+", version), (
        f"expected a bare MAJOR.MINOR.PATCH version, got {version!r}. "
        "The build workflow interpolates this into a tag and a 7z filename."
    )


def test_frontend_package_json_matches():
    pkg = json.loads((REPO_ROOT / "frontend" / "package.json").read_text(encoding="utf-8"))
    assert pkg["version"] == _expected_version(), (
        "frontend/package.json is out of sync with version.py"
    )


def test_tauri_conf_matches():
    conf_path = REPO_ROOT / "frontend" / "src-tauri" / "tauri.conf.json"
    conf = json.loads(conf_path.read_text(encoding="utf-8"))
    assert conf["version"] == _expected_version(), (
        "tauri.conf.json is out of sync with version.py — the installer would "
        "advertise a different version than the app reports at /health"
    )


def test_cargo_toml_matches():
    cargo = (REPO_ROOT / "frontend" / "src-tauri" / "Cargo.toml").read_text(encoding="utf-8")
    # The crate version is the first `version = "..."` under [package]; later
    # `version` keys belong to dependency tables.
    match = re.search(r'^\[package\](.*?)^\[', cargo, re.M | re.S)
    assert match, "Cargo.toml has no [package] section"
    version_match = re.search(r'^version\s*=\s*"([^"]+)"', match.group(1), re.M)
    assert version_match, "Cargo.toml [package] has no version"
    assert version_match.group(1) == _expected_version(), (
        "frontend/src-tauri/Cargo.toml is out of sync with version.py"
    )


def test_cargo_lock_matches_cargo_toml():
    """A stale lock version makes `cargo build --locked` fail in CI."""
    lock_path = REPO_ROOT / "frontend" / "src-tauri" / "Cargo.lock"
    if not lock_path.exists():
        pytest.skip("Cargo.lock not present")
    lock = lock_path.read_text(encoding="utf-8")
    match = re.search(
        r'^name = "video-highlighter"\nversion = "([^"]+)"', lock, re.M
    )
    assert match, "Cargo.lock has no video-highlighter package entry"
    assert match.group(1) == _expected_version(), (
        "Cargo.lock records a stale crate version; run `cargo update -p "
        "video-highlighter` or edit it to match version.py"
    )
