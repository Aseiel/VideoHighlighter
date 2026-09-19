"""Read, verify and diff release manifests.

This is the half of the updater that decides *what* would change. It downloads
nothing and writes nothing — it turns "here is a signed list of the files in
release X" plus "here is what is on disk" into a plan: these files must be
fetched, these must be deleted, that many bytes.

Two rules do the safety work:

**Only the signature is trusted.** ``verify_manifest`` checks an Ed25519
signature over the exact manifest bytes against a key embedded at build time.
An unsigned or mis-signed manifest is not "probably fine" — it is the exact
shape an attack takes, since acting on a manifest means replacing executables.
Verification fails closed, including when no key has been embedded yet.

**Only what we shipped is ever deleted.** A file is removed only if the
*installed* manifest listed it and the new one does not. Anything else on disk
is left alone, which is what keeps a user's imported models, edited config and
license files safe — none of them were ever in a manifest.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Optional

MANIFEST_FORMAT = 1
MANIFEST_FILENAME = "manifest.json"
SIGNATURE_FILENAME = "manifest.json.sig"

# Ed25519 public key for release manifests, filled in by
# ``tools/build_manifest.py keygen --update-module``. Deliberately NOT the
# license signing key: a leaked release key must not also mint licenses.
# Empty means "no releases can be verified" — every manifest is rejected.
RELEASE_PUBLIC_KEY_HEX = ""

_CHUNK = 1024 * 1024


def hash_file(path: str) -> str:
    """SHA-256 of a file, read in chunks so a multi-GB file is not loaded."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(_CHUNK)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Signature
# ---------------------------------------------------------------------------

def _b64url_decode(text: str) -> bytes:
    import base64
    padding = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + padding)


def verify_manifest(raw: bytes, signature: str) -> Optional[dict]:
    """The parsed manifest if the signature is valid, else ``None``.

    ``raw`` must be the bytes exactly as downloaded — re-encoding the JSON
    before verifying would change what is being checked and defeat the point.
    """
    if not RELEASE_PUBLIC_KEY_HEX:
        print("update_manifest: no release public key embedded; refusing.")
        return None
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PublicKey,
        )

        key = Ed25519PublicKey.from_public_bytes(
            bytes.fromhex(RELEASE_PUBLIC_KEY_HEX))
        try:
            key.verify(_b64url_decode((signature or "").strip()), raw)
        except InvalidSignature:
            print("update_manifest: signature does not match; refusing.")
            return None
    except Exception as exc:
        print(f"update_manifest: cannot verify ({type(exc).__name__}: {exc})")
        return None

    try:
        manifest = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        print(f"update_manifest: signed but unreadable ({exc})")
        return None

    if not isinstance(manifest, dict) or manifest.get("format") != MANIFEST_FORMAT:
        print("update_manifest: unsupported manifest format; refusing.")
        return None
    if not isinstance(manifest.get("files"), list):
        print("update_manifest: manifest has no file list; refusing.")
        return None
    for entry in manifest["files"]:
        if not isinstance(entry, dict) or not is_safe_relpath(entry.get("path")):
            print(f"update_manifest: unsafe path in manifest: {entry!r}")
            return None
    return manifest


def is_safe_relpath(path) -> bool:
    """Whether ``path`` may be joined onto the install root.

    A manifest entry names a file to write, so a path escaping the install
    directory is a write-anywhere primitive. Signed manifests should never
    contain one; this refuses regardless, because a check that only runs when
    something has already gone wrong is the one worth having.
    """
    if not isinstance(path, str) or not path or path.strip() != path:
        return False
    if path.startswith("/") or path.startswith("\\"):
        return False
    if ":" in path:                      # C:/..., or an NTFS alternate stream
        return False
    parts = path.replace("\\", "/").split("/")
    if any(part in ("", ".", "..") for part in parts):
        return False
    return True


def local_path(root: str, relative: str) -> str:
    """Absolute path for a manifest entry inside ``root``."""
    return os.path.join(root, relative.replace("/", os.sep))


# ---------------------------------------------------------------------------
# The installed manifest
# ---------------------------------------------------------------------------

def installed_manifest_path(root: str) -> str:
    return os.path.join(root, MANIFEST_FILENAME)


def load_installed_manifest(root: str) -> Optional[dict]:
    """The manifest that shipped with the running install.

    Unsigned on purpose: it describes what is already on this disk, so it is a
    record rather than a claim. Its only job is to say which files were ours,
    and the update it authorises is bounded by the *new* manifest, which is
    signed.
    """
    try:
        with open(installed_manifest_path(root), "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, ValueError):
        return None
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), list):
        return None
    return manifest


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

@dataclass
class UpdatePlan:
    """What applying a release would do. Nothing here has happened yet."""

    version: str = ""
    download: list = field(default_factory=list)   # manifest entries to fetch
    delete: list = field(default_factory=list)     # relative paths to remove
    unchanged: int = 0

    @property
    def download_bytes(self) -> int:
        return sum(int(entry.get("size", 0)) for entry in self.download)

    @property
    def is_empty(self) -> bool:
        return not self.download and not self.delete

    def summary(self) -> str:
        if self.is_empty:
            return "Already up to date."
        mb = self.download_bytes / (1024 * 1024)
        bits = [f"{len(self.download)} file(s) to download ({mb:.1f} MB)"]
        if self.delete:
            bits.append(f"{len(self.delete)} to remove")
        bits.append(f"{self.unchanged} unchanged")
        return ", ".join(bits)


def plan_update(new_manifest: dict, root: str,
                installed: Optional[dict] = None,
                *, verify_unchanged: bool = False) -> UpdatePlan:
    """Diff a verified manifest against what is on disk.

    Speed matters here: re-hashing several GB at every check would make the
    updater feel broken. So a file is taken as current when the installed
    manifest agrees with the new one *and* the file is still present. Files the
    two manifests disagree about are hashed, which means a partially applied
    update resumes instead of restarting. ``verify_unchanged=True`` hashes
    everything, for a "repair this install" action.
    """
    installed = installed if installed is not None else (load_installed_manifest(root) or {})
    installed_hashes = {
        entry.get("path"): entry.get("sha256")
        for entry in installed.get("files", [])
        if isinstance(entry, dict)
    }

    plan = UpdatePlan(version=str(new_manifest.get("version", "")))
    new_paths = set()

    for entry in new_manifest.get("files", []):
        relative = entry.get("path")
        new_paths.add(relative)
        absolute = local_path(root, relative)

        if not os.path.exists(absolute):
            plan.download.append(entry)
            continue

        if not verify_unchanged and installed_hashes.get(relative) == entry.get("sha256"):
            plan.unchanged += 1
            continue

        try:
            current = hash_file(absolute)
        except OSError:
            plan.download.append(entry)
            continue

        if current == entry.get("sha256"):
            plan.unchanged += 1
        else:
            plan.download.append(entry)

    # Only ever remove files a previous release of ours put there. A user's
    # imported models, edited config and license file were never in a manifest,
    # so they can never appear here.
    for relative in installed_hashes:
        if relative not in new_paths and is_safe_relpath(relative):
            if os.path.exists(local_path(root, relative)):
                plan.delete.append(relative)

    return plan
