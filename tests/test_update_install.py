"""Tests for the whole-update orchestration.

Everything is injected — the manifest fetch, the file opener, the signing key —
so a full install runs in a tmp dir with no network. The cases here are the
ones where a user is watching: an unverifiable release, a dead connection
halfway, a cancel.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from modules import update_install, update_manifest as um

BASE = "https://updates.example/bucket"
MANIFEST_URL = BASE + "/manifest.json"


@pytest.fixture
def key(monkeypatch):
    private = Ed25519PrivateKey.generate()
    monkeypatch.setattr(um, "RELEASE_PUBLIC_KEY_HEX",
                        private.public_key().public_bytes(
                            Encoding.Raw, PublicFormat.Raw).hex())
    return private


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _release(files: dict, version="0.9.1"):
    """``{relative_path: bytes}`` -> (manifest_bytes, blobs_by_digest)."""
    manifest = {
        "format": um.MANIFEST_FORMAT,
        "version": version,
        "edition": "Pro",
        "base_url": BASE,
        "files": [{"path": p, "size": len(d), "sha256": _sha(d)}
                  for p, d in files.items()],
    }
    return json.dumps(manifest).encode(), {_sha(d): d for d in files.values()}


def _fetcher(raw, private, *, break_signature=False):
    signature = private.sign(raw)
    if break_signature:
        signature = Ed25519PrivateKey.generate().sign(raw)
    encoded = base64.urlsafe_b64encode(signature).decode().rstrip("=")

    def fetch(url):
        if url.endswith(".sig"):
            return encoded.encode()
        return raw

    return fetch


def _opener(blobs, fail_after=None):
    served = {"n": 0}

    class _R(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *e):
            self.close()
            return False

    def opener(url, headers):
        digest = url.rsplit("/", 1)[-1]
        served["n"] += 1
        if fail_after is not None and served["n"] > fail_after:
            raise OSError("connection lost")
        return _R(blobs[digest])

    return opener


def _install(root, relative, data: bytes):
    path = os.path.join(str(root), relative.replace("/", os.sep))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(data)


# --- the happy path --------------------------------------------------------

def test_full_install(tmp_path, key):
    _install(tmp_path, "app.exe", b"v1")
    _install(tmp_path, "keep.bin", b"same")
    raw, blobs = _release({"app.exe": b"v2", "keep.bin": b"same"})

    result = update_install.install_update(
        MANIFEST_URL, str(tmp_path),
        fetch=_fetcher(raw, key), opener=_opener(blobs))

    assert result.ok and result.restart_required
    assert result.version == "0.9.1"
    assert (tmp_path / "app.exe").read_bytes() == b"v2"
    assert not os.path.exists(update_install.staging_dir(str(tmp_path))), \
        "staging should be cleaned up after a successful install"


def test_up_to_date_is_not_an_error(tmp_path, key):
    _install(tmp_path, "app.exe", b"v1")
    raw, blobs = _release({"app.exe": b"v1"})
    # An installed manifest recording the same state.
    with open(os.path.join(tmp_path, "manifest.json"), "wb") as handle:
        handle.write(raw)

    result = update_install.install_update(
        MANIFEST_URL, str(tmp_path), fetch=_fetcher(raw, key),
        opener=_opener(blobs))
    assert result.ok
    assert result.message == "Already up to date."
    assert not result.restart_required


def test_only_changed_files_are_fetched(tmp_path, key):
    _install(tmp_path, "app.exe", b"v1")
    _install(tmp_path, "big.bin", b"H" * 10_000)
    raw, blobs = _release({"app.exe": b"v2", "big.bin": b"H" * 10_000})

    result = update_install.install_update(
        MANIFEST_URL, str(tmp_path), fetch=_fetcher(raw, key),
        opener=_opener(blobs))
    assert result.ok
    assert result.downloaded_bytes == len(b"v2"), "the big file must be skipped"


# --- refusing bad releases -------------------------------------------------

def test_unverifiable_release_is_refused_and_nothing_changes(tmp_path, key):
    _install(tmp_path, "app.exe", b"v1")
    raw, blobs = _release({"app.exe": b"malicious"})

    result = update_install.install_update(
        MANIFEST_URL, str(tmp_path),
        fetch=_fetcher(raw, key, break_signature=True), opener=_opener(blobs))

    assert not result.ok
    assert "could not be verified" in result.message
    assert (tmp_path / "app.exe").read_bytes() == b"v1", "install untouched"


def test_unreachable_server_is_reported(tmp_path, key):
    def fetch(url):
        raise OSError("no route to host")

    result = update_install.install_update(
        MANIFEST_URL, str(tmp_path), fetch=fetch)
    assert not result.ok
    assert "Could not reach" in result.message


def test_release_without_base_url_is_refused(tmp_path, key):
    manifest = {"format": um.MANIFEST_FORMAT, "version": "0.9.1",
                "base_url": "", "files": []}
    raw = json.dumps(manifest).encode()
    result = update_install.install_update(
        MANIFEST_URL, str(tmp_path), fetch=_fetcher(raw, key))
    assert not result.ok
    assert "individual files" in result.message


# --- interruptions ---------------------------------------------------------

def test_a_dropped_connection_leaves_the_install_alone(tmp_path, key):
    _install(tmp_path, "a.bin", b"a-v1")
    _install(tmp_path, "b.bin", b"b-v1")
    raw, blobs = _release({"a.bin": b"a-v2", "b.bin": b"b-v2"})

    result = update_install.install_update(
        MANIFEST_URL, str(tmp_path), fetch=_fetcher(raw, key),
        opener=_opener(blobs, fail_after=1))

    assert not result.ok
    assert "failed to download" in result.message
    assert (tmp_path / "a.bin").read_bytes() == b"a-v1"
    assert (tmp_path / "b.bin").read_bytes() == b"b-v1"


def test_an_interrupted_download_resumes(tmp_path, key):
    _install(tmp_path, "a.bin", b"a-v1")
    _install(tmp_path, "b.bin", b"b-v1")
    raw, blobs = _release({"a.bin": b"a-v2", "b.bin": b"b-v2"})

    first = update_install.install_update(
        MANIFEST_URL, str(tmp_path), fetch=_fetcher(raw, key),
        opener=_opener(blobs, fail_after=1))
    assert not first.ok
    assert update_install.pending_bytes(str(tmp_path)) > 0, "keep what arrived"

    second = update_install.install_update(
        MANIFEST_URL, str(tmp_path), fetch=_fetcher(raw, key),
        opener=_opener(blobs))
    assert second.ok
    assert (tmp_path / "a.bin").read_bytes() == b"a-v2"
    assert (tmp_path / "b.bin").read_bytes() == b"b-v2"


def test_cancel_keeps_partial_work(tmp_path, key):
    _install(tmp_path, "a.bin", b"a-v1")
    raw, blobs = _release({"a.bin": b"a-v2"})

    result = update_install.install_update(
        MANIFEST_URL, str(tmp_path), fetch=_fetcher(raw, key),
        opener=_opener(blobs), should_cancel=lambda: True)
    assert result.cancelled
    assert (tmp_path / "a.bin").read_bytes() == b"a-v1"


def test_discard_staged_clears_it(tmp_path, key):
    os.makedirs(update_install.staging_dir(str(tmp_path)))
    with open(os.path.join(update_install.staging_dir(str(tmp_path)), "x"), "wb") as h:
        h.write(b"partial")
    assert update_install.pending_bytes(str(tmp_path)) == 7
    update_install.discard_staged(str(tmp_path))
    assert update_install.pending_bytes(str(tmp_path)) == 0


# --- progress --------------------------------------------------------------

def test_progress_phases_are_reported(tmp_path, key):
    _install(tmp_path, "app.exe", b"v1")
    raw, blobs = _release({"app.exe": b"v2"})
    phases = []
    update_install.install_update(
        MANIFEST_URL, str(tmp_path), fetch=_fetcher(raw, key),
        opener=_opener(blobs),
        progress=lambda phase, done, total, detail: phases.append(phase))

    assert update_install.CHECKING in phases
    assert update_install.DOWNLOADING in phases
    assert update_install.INSTALLING in phases
