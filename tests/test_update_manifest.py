"""Tests for release manifest verification and the update diff.

Signing happens with a throwaway key generated per test and monkeypatched into
the module, so nothing here touches the real release key. What's pinned down is
what a bad manifest can and cannot make the updater do — because acting on a
manifest means replacing executables on someone's machine.
"""
from __future__ import annotations

import base64
import json
import os

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from modules import update_manifest as um


@pytest.fixture
def signer(monkeypatch):
    """A real Ed25519 keypair whose public half the module trusts."""
    private = Ed25519PrivateKey.generate()
    public_hex = private.public_key().public_bytes(
        Encoding.Raw, PublicFormat.Raw).hex()
    monkeypatch.setattr(um, "RELEASE_PUBLIC_KEY_HEX", public_hex)

    def sign(raw: bytes) -> str:
        return base64.urlsafe_b64encode(private.sign(raw)).decode().rstrip("=")

    return sign


def _manifest(files, version="0.9.1"):
    return {
        "format": um.MANIFEST_FORMAT,
        "version": version,
        "edition": "Pro",
        "files": files,
    }


def _entry(path, content):
    import hashlib
    return {"path": path, "size": len(content),
            "sha256": hashlib.sha256(content).hexdigest()}


def _raw(manifest) -> bytes:
    return json.dumps(manifest).encode("utf-8")


def _write(root, relative, content: bytes):
    absolute = os.path.join(str(root), relative.replace("/", os.sep))
    os.makedirs(os.path.dirname(absolute), exist_ok=True)
    with open(absolute, "wb") as handle:
        handle.write(content)
    return absolute


# --- signature -------------------------------------------------------------

def test_valid_signature_is_accepted(signer):
    raw = _raw(_manifest([_entry("app.exe", b"hello")]))
    assert um.verify_manifest(raw, signer(raw)) is not None


def test_tampered_manifest_is_rejected(signer):
    raw = _raw(_manifest([_entry("app.exe", b"hello")]))
    signature = signer(raw)
    tampered = raw.replace(b"0.9.1", b"9.9.9")
    assert um.verify_manifest(tampered, signature) is None


def test_signature_from_another_key_is_rejected(signer):
    raw = _raw(_manifest([_entry("app.exe", b"hello")]))
    attacker = Ed25519PrivateKey.generate()
    forged = base64.urlsafe_b64encode(attacker.sign(raw)).decode().rstrip("=")
    assert um.verify_manifest(raw, forged) is None


def test_garbage_signature_is_rejected(signer):
    raw = _raw(_manifest([_entry("app.exe", b"hello")]))
    assert um.verify_manifest(raw, "not-a-signature") is None


def test_no_embedded_key_rejects_everything(monkeypatch):
    monkeypatch.setattr(um, "RELEASE_PUBLIC_KEY_HEX", "")
    raw = _raw(_manifest([_entry("app.exe", b"hello")]))
    assert um.verify_manifest(raw, "anything") is None, "must fail closed"


def test_unknown_format_is_rejected(signer):
    manifest = _manifest([_entry("app.exe", b"hello")])
    manifest["format"] = 99
    raw = _raw(manifest)
    assert um.verify_manifest(raw, signer(raw)) is None


# --- path safety -----------------------------------------------------------

@pytest.mark.parametrize("path", [
    "../outside.txt",
    "a/../../outside.txt",
    "/etc/passwd",
    "\\windows\\system32\\evil.dll",
    "C:/Windows/evil.dll",
    "app.exe:stream",
    "",
    " leading.txt",
    None,
    123,
])
def test_unsafe_paths_are_refused(path):
    assert um.is_safe_relpath(path) is False


@pytest.mark.parametrize("path", [
    "app.exe",
    "_internal/modules/pipeline.py",
    "_internal/models/yolox/model.onnx",
])
def test_normal_paths_are_allowed(path):
    assert um.is_safe_relpath(path) is True


def test_a_signed_manifest_with_a_traversal_path_is_still_rejected(signer):
    """Even correctly signed: a compromised key must not become file-write."""
    raw = _raw(_manifest([{"path": "../evil.exe", "size": 1, "sha256": "ab"}]))
    assert um.verify_manifest(raw, signer(raw)) is None


# --- the diff --------------------------------------------------------------

def test_only_changed_files_are_downloaded(tmp_path):
    _write(tmp_path, "same.txt", b"unchanged")
    _write(tmp_path, "changed.txt", b"old content")

    installed = _manifest([_entry("same.txt", b"unchanged"),
                           _entry("changed.txt", b"old content")], "0.9.0")
    new = _manifest([_entry("same.txt", b"unchanged"),
                     _entry("changed.txt", b"new content")])

    plan = um.plan_update(new, str(tmp_path), installed)
    assert [e["path"] for e in plan.download] == ["changed.txt"]
    assert plan.unchanged == 1
    assert plan.delete == []


def test_new_files_are_downloaded(tmp_path):
    _write(tmp_path, "old.txt", b"x")
    installed = _manifest([_entry("old.txt", b"x")], "0.9.0")
    new = _manifest([_entry("old.txt", b"x"), _entry("added.txt", b"y")])

    plan = um.plan_update(new, str(tmp_path), installed)
    assert [e["path"] for e in plan.download] == ["added.txt"]


def test_removed_files_are_deleted(tmp_path):
    _write(tmp_path, "gone.txt", b"x")
    _write(tmp_path, "stays.txt", b"y")
    installed = _manifest([_entry("gone.txt", b"x"), _entry("stays.txt", b"y")], "0.9.0")
    new = _manifest([_entry("stays.txt", b"y")])

    plan = um.plan_update(new, str(tmp_path), installed)
    assert plan.delete == ["gone.txt"]


def test_user_files_are_never_deleted(tmp_path):
    """The rule that protects imported models, edited config and licenses."""
    _write(tmp_path, "shipped.txt", b"x")
    _write(tmp_path, "config.yaml", b"user edited this")
    _write(tmp_path, "license.key", b"VHPRO-1.xxx")
    _write(tmp_path, "_internal/models/custom/mine.onnx", b"user model")

    installed = _manifest([_entry("shipped.txt", b"x")], "0.9.0")
    new = _manifest([_entry("shipped.txt", b"x")])

    plan = um.plan_update(new, str(tmp_path), installed)
    assert plan.delete == []
    assert os.path.exists(os.path.join(tmp_path, "config.yaml"))


def test_missing_file_is_redownloaded_even_if_manifests_agree(tmp_path):
    installed = _manifest([_entry("deleted.txt", b"x")], "0.9.0")
    new = _manifest([_entry("deleted.txt", b"x")])
    plan = um.plan_update(new, str(tmp_path), installed)
    assert [e["path"] for e in plan.download] == ["deleted.txt"]


def test_a_partly_applied_update_resumes(tmp_path):
    """A file already matching the NEW hash is not fetched again."""
    _write(tmp_path, "a.txt", b"new content")
    installed = _manifest([_entry("a.txt", b"old content")], "0.9.0")
    new = _manifest([_entry("a.txt", b"new content")])

    plan = um.plan_update(new, str(tmp_path), installed)
    assert plan.download == []
    assert plan.unchanged == 1


def test_no_installed_manifest_downloads_everything(tmp_path):
    _write(tmp_path, "a.txt", b"whatever")
    new = _manifest([_entry("a.txt", b"different"), _entry("b.txt", b"new")])
    plan = um.plan_update(new, str(tmp_path), {})
    assert len(plan.download) == 2


def test_verify_unchanged_catches_a_corrupted_file(tmp_path):
    """Repair mode: manifests agree, but the bytes on disk do not."""
    _write(tmp_path, "a.txt", b"CORRUPTED")
    installed = _manifest([_entry("a.txt", b"good")], "0.9.0")
    new = _manifest([_entry("a.txt", b"good")])

    fast = um.plan_update(new, str(tmp_path), installed)
    assert fast.download == [], "the fast path trusts the manifests"

    repair = um.plan_update(new, str(tmp_path), installed, verify_unchanged=True)
    assert [e["path"] for e in repair.download] == ["a.txt"]


def test_plan_summary_reports_size(tmp_path):
    new = _manifest([{"path": "big.bin", "size": 5 * 1024 * 1024, "sha256": "ff"}])
    plan = um.plan_update(new, str(tmp_path), {})
    assert plan.download_bytes == 5 * 1024 * 1024
    assert "5.0 MB" in plan.summary()
    assert not plan.is_empty


def test_identical_release_is_a_no_op(tmp_path):
    _write(tmp_path, "a.txt", b"x")
    installed = _manifest([_entry("a.txt", b"x")], "0.9.0")
    plan = um.plan_update(_manifest([_entry("a.txt", b"x")]), str(tmp_path), installed)
    assert plan.is_empty
    assert plan.summary() == "Already up to date."
