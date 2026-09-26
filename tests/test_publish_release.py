"""Tests for the release tooling: generate, prepare, sign, channel.

The whole path a release takes to the update host, offline, in a tmp dir: a
fake bundle is hashed into a manifest, laid out content-addressed, signed with
a throwaway key, and announced in a channel file — then an install is updated
from exactly those files, so the layout the tools write is the one the app
reads.
"""
from __future__ import annotations

import io
import json
import os

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import (
    Encoding, NoEncryption, PrivateFormat, PublicFormat,
)

from modules.update import update_download, update_install, update_manifest as um
from tools import build_manifest, publish_release

BASE = "https://updates.example/vh"


@pytest.fixture
def key(tmp_path, monkeypatch):
    private = Ed25519PrivateKey.generate()
    monkeypatch.setattr(um, "RELEASE_PUBLIC_KEY_HEX", private.public_key().public_bytes(
        Encoding.Raw, PublicFormat.Raw).hex())
    path = tmp_path / "release.pem"
    path.write_bytes(private.private_bytes(
        Encoding.PEM, PrivateFormat.PKCS8, NoEncryption()))
    return str(path)


def _bundle(root, files: dict):
    for relative, data in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def _generate(root, version="1.2.0", **extra):
    argv = ["generate", "--root", str(root), "--version", version,
            "--edition", "Free", "--base-url", BASE]
    for flag, value in extra.items():
        argv += [f"--{flag.replace('_', '-')}", value]
    assert build_manifest.main(argv) == 0
    return json.loads((root / "manifest.json").read_text())


# --- generate ---------------------------------------------------------------

def test_generate_stamps_platform_and_skips_what_is_not_the_release(tmp_path):
    dist = tmp_path / "dist"
    _bundle(dist, {
        "VideoHighlighter.exe": b"exe",
        "_internal/app.pyd": b"pyd",
        # Installed beside the app after the build, never part of it:
        "packs/torch-cpu/site-packages/torch/__init__.py": b"torch",
        ".update-old/VideoHighlighter.exe": b"old",
    })
    manifest = _generate(dist, min_version="1.1.0")
    assert manifest["platform"] == "windows"
    assert manifest["min_version"] == "1.1.0"
    assert sorted(e["path"] for e in manifest["files"]) == [
        "VideoHighlighter.exe", "_internal/app.pyd"]


def test_keygen_patches_the_module_where_it_now_lives():
    # The module moved into modules/update/; keygen --update-module used to
    # look for it at the old path and fail.
    assert os.path.exists(build_manifest._MODULE_PATH)
    with open(build_manifest._MODULE_PATH, encoding="utf-8") as handle:
        assert "RELEASE_PUBLIC_KEY_HEX = " in handle.read()


# --- prepare ----------------------------------------------------------------

def test_prepare_lays_out_blobs_and_the_release_manifest(tmp_path):
    dist = tmp_path / "dist"
    _bundle(dist, {"a.exe": b"same", "b/copy.exe": b"same", "c.dll": b"other"})
    manifest = _generate(dist)
    out = tmp_path / "publish"

    assert publish_release.main([
        "prepare", "--root", str(dist), "--out", str(out), "--allow-unsigned"]) == 0

    blobs = sorted(os.listdir(out / "files"))
    assert blobs == sorted({e["sha256"] for e in manifest["files"]})   # deduplicated
    staged = out / "releases" / "free" / "windows" / "1.2.0" / "manifest.json"
    assert staged.read_bytes() == (dist / "manifest.json").read_bytes()


def test_prepare_refuses_unsigned_unless_told(tmp_path):
    dist = tmp_path / "dist"
    _bundle(dist, {"a.exe": b"x"})
    _generate(dist)
    assert publish_release.main([
        "prepare", "--root", str(dist), "--out", str(tmp_path / "p")]) == 1


# --- sign -------------------------------------------------------------------

def test_sign_produces_what_the_app_accepts(tmp_path, key):
    dist = tmp_path / "dist"
    _bundle(dist, {"a.exe": b"x"})
    _generate(dist)
    out = tmp_path / "m.sig"
    assert publish_release.main([
        "sign", "--manifest", str(dist / "manifest.json"), "--key", key,
        "--out", str(out)]) == 0
    raw = (dist / "manifest.json").read_bytes()
    assert um.verify_manifest(raw, out.read_text().strip()) is not None


def test_sign_refuses_a_key_the_app_does_not_embed(tmp_path, key, monkeypatch):
    dist = tmp_path / "dist"
    _bundle(dist, {"a.exe": b"x"})
    _generate(dist)
    monkeypatch.setattr(um, "RELEASE_PUBLIC_KEY_HEX", "00" * 32)
    assert publish_release.main([
        "sign", "--manifest", str(dist / "manifest.json"), "--key", key,
        "--out", str(tmp_path / "m.sig")]) == 1


def test_sign_refuses_a_manifest_for_another_version(tmp_path, key):
    dist = tmp_path / "dist"
    _bundle(dist, {"a.exe": b"x"})
    _generate(dist, version="1.2.0")
    assert publish_release.main([
        "sign", "--manifest", str(dist / "manifest.json"), "--version", "1.3.0",
        "--key", key, "--out", str(tmp_path / "m.sig")]) == 1


# --- channel ----------------------------------------------------------------

def test_channel_names_this_platforms_manifest(tmp_path):
    dist = tmp_path / "dist"
    _bundle(dist, {"a.exe": b"x"})
    _generate(dist, notes="Faster exports.")
    out = tmp_path / "free.json"
    assert publish_release.main([
        "channel", "--manifest", str(dist / "manifest.json"), "--out", str(out)]) == 0
    channel = json.loads(out.read_text())
    assert channel["version"] == "1.2.0"
    assert channel["notes"] == "Faster exports."
    assert channel["manifests"] == {
        "windows": f"{BASE}/releases/free/windows/1.2.0/manifest.json"}


def test_channel_keeps_other_platforms_of_the_same_version_only():
    manifest = {"version": "1.2.0", "platform": "windows"}
    same = {"version": "1.2.0", "manifests": {"macos": "m"}}
    older = {"version": "1.1.0", "manifests": {"macos": "m"}}
    assert publish_release.build_channel(manifest, "w", previous=same)["manifests"] == {
        "macos": "m", "windows": "w"}
    assert publish_release.build_channel(manifest, "w", previous=older)["manifests"] == {
        "windows": "w"}


# --- the round trip ---------------------------------------------------------

def test_an_install_updates_from_exactly_what_the_tools_publish(tmp_path, key, monkeypatch):
    # 1.1.0 is installed; 1.2.0 changes one file, adds one, drops one.
    installed = tmp_path / "install"
    _bundle(installed, {"app.exe": b"v1", "keep.dll": b"same", "gone.py": b"old"})
    _generate(installed, version="1.1.0")

    dist = tmp_path / "dist"
    _bundle(dist, {"app.exe": b"v2", "keep.dll": b"same", "new.py": b"new"})
    _generate(dist, version="1.2.0")

    host = tmp_path / "host"
    assert publish_release.main([
        "prepare", "--root", str(dist), "--out", str(host), "--allow-unsigned"]) == 0
    manifest_path = host / "releases" / "free" / "windows" / "1.2.0" / "manifest.json"
    sig_path = str(manifest_path) + ".sig"
    assert publish_release.main([
        "sign", "--manifest", str(manifest_path), "--key", key, "--out", sig_path]) == 0

    def fetch(url):
        return (host / url[len(BASE) + 1:]).read_bytes()

    class _R(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.close()
            return False

    requested = []

    def opener(url, headers):
        requested.append(url)
        return _R((host / url[len(BASE) + 1:]).read_bytes())

    monkeypatch.setattr(update_install, "_running", lambda: ("1.1.0", "Free", "windows"))
    monkeypatch.setattr(update_download, "RETRY_DELAYS", (0, 0))
    result = update_install.install_update(
        f"{BASE}/releases/free/windows/1.2.0/manifest.json", str(installed),
        fetch=fetch, opener=opener)

    assert result.ok, result.message
    assert (installed / "app.exe").read_bytes() == b"v2"
    assert (installed / "new.py").read_bytes() == b"new"
    assert not (installed / "gone.py").exists()
    assert len(requested) == 2          # keep.dll was not downloaded again


# --- check (what publish-update.yaml runs before anything is visible) -------

def _staged(tmp_path, key, version="1.2.0"):
    dist = tmp_path / "dist"
    _bundle(dist, {"a.exe": b"x", "b.dll": b"yy"})
    manifest = _generate(dist, version=version)
    sig = tmp_path / "manifest.json.sig"
    assert publish_release.main([
        "sign", "--manifest", str(dist / "manifest.json"), "--key", key,
        "--out", str(sig)]) == 0
    listing = tmp_path / "listing.json"
    listing.write_text(json.dumps(
        [[f"files/{e['sha256']}", e["size"]] for e in manifest["files"]]))
    return dist / "manifest.json", sig, listing, manifest


def _check(manifest, sig, listing, version="1.2.0", base=BASE):
    return publish_release.main([
        "check", "--manifest", str(manifest), "--sig", str(sig),
        "--version", version, "--edition", "Free", "--base-url", base,
        "--listing", str(listing)])


def test_check_passes_a_signed_fully_staged_release(tmp_path, key):
    manifest, sig, listing, _ = _staged(tmp_path, key)
    assert _check(manifest, sig, listing) == 0


def test_check_refuses_a_bad_signature(tmp_path, key):
    manifest, sig, listing, _ = _staged(tmp_path, key)
    sig.write_text("A" * 86)
    assert _check(manifest, sig, listing) == 1


def test_check_refuses_a_missing_or_short_blob(tmp_path, key):
    manifest, sig, listing, files = _staged(tmp_path, key)
    rows = json.loads(listing.read_text())
    listing.write_text(json.dumps(rows[:1]))
    assert _check(manifest, sig, listing) == 1
    listing.write_text(json.dumps([[k, s + 1] for k, s in rows]))
    assert _check(manifest, sig, listing) == 1


def test_check_refuses_another_version_or_host(tmp_path, key):
    manifest, sig, listing, _ = _staged(tmp_path, key)
    assert _check(manifest, sig, listing, version="1.2.1") == 1
    assert _check(manifest, sig, listing, base="https://elsewhere.example") == 1


def test_the_channel_never_moves_backwards(tmp_path):
    dist = tmp_path / "dist"
    _bundle(dist, {"a.exe": b"x"})
    _generate(dist, version="1.2.0")
    previous = tmp_path / "previous.json"
    previous.write_text(json.dumps({"version": "1.3.0", "manifests": {}}))
    assert publish_release.main([
        "channel", "--manifest", str(dist / "manifest.json"),
        "--previous", str(previous), "--out", str(tmp_path / "c.json")]) == 1


def test_prefix_matches_what_prepare_writes(capsys):
    assert publish_release.main([
        "prefix", "--version", "1.2.0", "--edition", "Pro"]) == 0
    assert capsys.readouterr().out.strip() == "releases/pro/windows/1.2.0"


# --- compressed blobs ------------------------------------------------------------

def test_gzip_blobs_round_trip_and_are_reproducible(tmp_path, key, monkeypatch):
    import gzip

    installed = tmp_path / "install"
    _bundle(installed, {"app.exe": b"v1" * 1000})
    _generate(installed, version="1.1.0")

    dist = tmp_path / "dist"
    payload = b"new build " * 5000                    # compresses well
    _bundle(dist, {"app.exe": payload})
    manifest = _generate(dist, version="1.2.0", compression="gzip")
    assert manifest["compression"] == "gzip"

    host = tmp_path / "host"
    assert publish_release.main(["prepare", "--root", str(dist), "--out", str(host),
                                 "--allow-unsigned"]) == 0
    blob = host / "files" / (manifest["files"][0]["sha256"] + ".gz")
    assert blob.exists() and blob.stat().st_size < len(payload) // 10
    assert gzip.decompress(blob.read_bytes()) == payload
    # Staged again from scratch: the same bytes, so a size-only sync is right.
    again = tmp_path / "host2"
    publish_release.main(["prepare", "--root", str(dist), "--out", str(again),
                          "--allow-unsigned"])
    assert (again / "files" / blob.name).read_bytes() == blob.read_bytes()

    manifest_path = host / "releases" / "free" / "windows" / "1.2.0" / "manifest.json"
    sig = str(manifest_path) + ".sig"
    assert publish_release.main(["sign", "--manifest", str(manifest_path), "--key", key,
                                 "--out", sig]) == 0

    class _R(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.close()
            return False

    fetched = []

    def opener(url, headers):
        fetched.append(url)
        return _R((host / url[len(BASE) + 1:]).read_bytes())

    monkeypatch.setattr(update_install, "_running", lambda: ("1.1.0", "Free", "windows"))
    monkeypatch.setattr(update_download, "RETRY_DELAYS", (0, 0))
    result = update_install.install_update(
        f"{BASE}/releases/free/windows/1.2.0/manifest.json", str(installed),
        fetch=lambda url: (host / url[len(BASE) + 1:]).read_bytes(), opener=opener)
    assert result.ok, result.message
    assert (installed / "app.exe").read_bytes() == payload
    assert fetched[0].endswith(".gz")

    # check: the listing holds .gz names.
    listing = tmp_path / "listing.json"
    listing.write_text(json.dumps([[f"files/{blob.name}", blob.stat().st_size]]))
    assert publish_release.main([
        "check", "--manifest", str(manifest_path), "--sig", sig, "--version", "1.2.0",
        "--edition", "Free", "--base-url", BASE, "--listing", str(listing)]) == 0


def test_a_cut_off_compressed_blob_fails_and_is_retried(tmp_path, monkeypatch):
    import gzip

    from modules.update.update_manifest import UpdatePlan

    data = b"payload " * 4000
    whole = gzip.compress(data)
    served = []

    class _R(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.close()
            return False

    def opener(url, headers):
        served.append(url)
        return _R(whole[: len(whole) // 2] if len(served) == 1 else whole)

    import hashlib
    plan = UpdatePlan(version="1")
    plan.download = [{"path": "a.bin", "size": len(data),
                      "sha256": hashlib.sha256(data).hexdigest()}]
    monkeypatch.setattr(update_download, "RETRY_DELAYS", (0, 0))
    result = update_download.download_plan(plan, BASE, str(tmp_path / "stage"),
                                           opener=opener, compression="gzip")
    assert result.ok and len(served) == 2
    assert (tmp_path / "stage" / "a.bin").read_bytes() == data
    assert result.bytes_done == len(data)


def test_a_manifest_naming_an_unknown_compression_is_refused(key):
    raw = json.dumps({"format": um.MANIFEST_FORMAT, "version": "1", "files": [],
                      "compression": "zstd"}).encode()
    from cryptography.hazmat.primitives.serialization import load_pem_private_key
    import base64
    with open(key, "rb") as fh:
        private = load_pem_private_key(fh.read(), password=None)
    sig = base64.urlsafe_b64encode(private.sign(raw)).decode().rstrip("=")
    assert um.verify_manifest(raw, sig) is None
