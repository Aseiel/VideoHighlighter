"""Run a complete update: fetch the release manifest, verify, download, apply.

The pieces each do one thing — ``update_manifest`` verifies and diffs,
``update_download`` fetches and hash-checks, ``update_apply`` swaps files. This
ties them into the single operation the UI actually invokes, so the GUI layer
holds no update logic and this can be tested without Qt or a network.

Order is deliberate and not rearrangeable:

1. verify the signature — nothing else may happen first;
2. diff against what is installed, so only changed files are fetched;
3. download to staging, verifying every file's hash there;
4. apply only once **every** file is present and correct.

Step 4 is what makes a dropped connection harmless: an interrupted download
leaves the install untouched, and the staged files are reused when the user
tries again.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional

from modules import update_apply, update_download, update_manifest

STAGING_DIRNAME = ".update-staging"

# Progress phases, so a caller can label a progress bar without parsing text.
CHECKING = "checking"
DOWNLOADING = "downloading"
INSTALLING = "installing"


@dataclass
class InstallResult:
    ok: bool = False
    message: str = ""
    version: str = ""
    restart_required: bool = False
    downloaded_bytes: int = 0
    cancelled: bool = False


def staging_dir(root: str) -> str:
    """Inside the install root on purpose: same volume, so applying an update
    is a rename rather than a copy of every downloaded file."""
    return os.path.join(root, STAGING_DIRNAME)


def _fetch_bytes(url: str) -> bytes:
    import urllib.request

    request = urllib.request.Request(
        url, headers={"Accept": "*/*"}, method="GET")
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def install_update(
    manifest_url: str,
    root: str,
    *,
    progress: Optional[Callable[[str, int, int, str], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
    fetch: Optional[Callable[[str], bytes]] = None,
    opener: Optional[Callable] = None,
) -> InstallResult:
    """Download and install the release described at ``manifest_url``.

    ``progress(phase, done, total, detail)`` is called throughout; ``phase`` is
    one of the module constants. Returns rather than raises — the caller is a
    button, and every failure here is something to show, not to crash on.
    """
    result = InstallResult()
    get = fetch or _fetch_bytes

    def report(phase, done=0, total=0, detail=""):
        if progress:
            progress(phase, done, total, detail)

    # 1. Fetch and verify --------------------------------------------------
    report(CHECKING, 0, 0, "Checking the release...")
    try:
        raw = get(manifest_url)
        signature = get(manifest_url + ".sig").decode("ascii").strip()
    except Exception as exc:
        result.message = f"Could not reach the update server ({exc})."
        return result

    manifest = update_manifest.verify_manifest(raw, signature)
    if manifest is None:
        # Deliberately blunt: this is either corruption or an attack, and in
        # both cases the answer is to stop and get the file from the vendor.
        result.message = ("This update could not be verified as genuine and "
                          "was not installed. Download it from your account "
                          "page instead.")
        return result

    result.version = str(manifest.get("version", ""))
    base_url = str(manifest.get("base_url") or "")
    if not base_url:
        result.message = "This release does not publish individual files."
        return result

    # 2. Diff ---------------------------------------------------------------
    report(CHECKING, 0, 0, "Working out what changed...")
    plan = update_manifest.plan_update(manifest, root)
    if plan.is_empty:
        result.ok = True
        result.message = "Already up to date."
        return result

    if should_cancel and should_cancel():
        result.cancelled = True
        result.message = "Cancelled."
        return result

    # 3. Download -----------------------------------------------------------
    staging = staging_dir(root)
    os.makedirs(staging, exist_ok=True)

    def on_bytes(done, total, path):
        report(DOWNLOADING, done, total, os.path.basename(path))

    downloaded = update_download.download_plan(
        plan, base_url, staging,
        progress=on_bytes, should_cancel=should_cancel, opener=opener)
    result.downloaded_bytes = downloaded.bytes_done

    if downloaded.cancelled:
        result.cancelled = True
        result.message = ("Cancelled. What was downloaded is kept, so trying "
                          "again resumes.")
        return result
    if not downloaded.ok:
        first = downloaded.failed[0]
        result.message = (f"{len(downloaded.failed)} file(s) failed to "
                          f"download ({first[0]}: {first[1]}). Nothing was "
                          "changed; try again.")
        return result

    # 4. Apply --------------------------------------------------------------
    report(INSTALLING, 0, 0, "Installing...")
    applied = update_apply.apply_update(
        root, staging, downloaded.staged, plan.delete, manifest_bytes=raw)
    if not applied.ok:
        result.message = (f"The update could not be installed ({applied.error}). "
                          "Your existing version was restored.")
        return result

    _cleanup(staging)
    result.ok = True
    result.restart_required = True
    result.message = f"Version {result.version} installed. Restart to use it."
    return result


def _cleanup(staging: str) -> None:
    import shutil
    shutil.rmtree(staging, ignore_errors=True)


def discard_staged(root: str) -> None:
    """Throw away a part-finished download (a "start over" action)."""
    _cleanup(staging_dir(root))


def pending_bytes(root: str) -> int:
    """How much of an interrupted download is still on disk."""
    path = staging_dir(root)
    if not os.path.isdir(path):
        return 0
    return update_download.staged_bytes(path)
