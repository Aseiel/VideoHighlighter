"""Fetch the files an update plan asks for, into a staging directory.

Nothing here touches the installed app. Files land in a staging folder and are
verified there; only ``update_apply`` moves them into place. That separation is
what makes a failed or cancelled download a non-event — the install is
untouched until every byte is present and checked.

Two rules:

**A file is not trusted until its hash matches.** Every download is verified
against the SHA-256 in the signed manifest before it counts as staged. A
truncated download, a proxy serving an error page as 200, or a tampered file
all fail the same check. The wrong bytes never reach the install directory.

**Interrupted work is kept.** Staged files that already match are skipped, so
cancelling and retrying resumes rather than starting over — which matters when
the payload is large and the connection is not.

Host-agnostic on purpose: the base URL and any auth headers are passed in, so
the same code serves a public release host or a gated one.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Optional

from modules.update_manifest import hash_file, is_safe_relpath, local_path

TIMEOUT_SECONDS = 30
_CHUNK = 256 * 1024


@dataclass
class DownloadResult:
    staged: list = field(default_factory=list)   # relative paths now in staging
    failed: list = field(default_factory=list)   # (relative path, reason)
    bytes_done: int = 0
    cancelled: bool = False

    @property
    def ok(self) -> bool:
        return not self.failed and not self.cancelled


def _default_opener(url: str, headers: dict):
    import urllib.request

    request = urllib.request.Request(url, headers=headers or {}, method="GET")
    return urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS)


def file_url(base_url: str, entry: dict, layout: str = "content") -> str:
    """Where to fetch one manifest entry from.

    ``content`` (the default) addresses files by their SHA-256 rather than by
    their path: ``<base>/files/<sha256>``. That has three consequences worth
    the indirection —

    * a file that did not change between releases is *already* on the host, so
      publishing a release uploads only genuinely new bytes (the multi-GB
      models get uploaded exactly once, ever);
    * an install can jump 0.9.0 -> 0.9.4 directly, because every blob any
      version ever referenced is still addressable;
    * blobs are immutable, so they can be cached forever and can never be
      silently swapped for different content under the same URL.

    ``path`` mirrors the install tree instead (``<base>/_internal/app.py``),
    for a host where a browsable layout matters more.
    """
    from urllib.parse import quote

    base = base_url.rstrip("/")
    if layout == "path":
        return base + "/" + quote(entry["path"])
    return base + "/files/" + quote(str(entry["sha256"]))


def download_plan(
    plan,
    base_url: str,
    staging_dir: str,
    *,
    layout: str = "content",
    headers: Optional[dict] = None,
    progress: Optional[Callable[[int, int, str], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
    opener: Optional[Callable] = None,
) -> DownloadResult:
    """Download every file in ``plan`` into ``staging_dir``.

    ``progress(bytes_done, bytes_total, current_path)`` is called as data
    arrives — often enough for a progress bar, not per chunk of every file.
    ``should_cancel()`` is polled between chunks so a user can stop a multi-GB
    download without waiting for it to finish.
    """
    result = DownloadResult()
    total = plan.download_bytes
    fetch = opener or _default_opener

    for entry in plan.download:
        relative = entry.get("path")
        if not is_safe_relpath(relative):
            result.failed.append((str(relative), "unsafe path"))
            continue

        if should_cancel and should_cancel():
            result.cancelled = True
            return result

        target = local_path(staging_dir, relative)
        expected = entry.get("sha256")

        # Already staged from an earlier attempt?
        if os.path.exists(target):
            try:
                if hash_file(target) == expected:
                    result.staged.append(relative)
                    result.bytes_done += int(entry.get("size", 0))
                    if progress:
                        progress(result.bytes_done, total, relative)
                    continue
            except OSError:
                pass
            try:
                os.remove(target)
            except OSError:
                pass

        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        partial = target + ".part"
        url = file_url(base_url, entry, layout)

        try:
            with fetch(url, headers or {}) as response:
                with open(partial, "wb") as handle:
                    while True:
                        if should_cancel and should_cancel():
                            result.cancelled = True
                            handle.close()
                            _remove(partial)
                            return result
                        block = response.read(_CHUNK)
                        if not block:
                            break
                        handle.write(block)
                        result.bytes_done += len(block)
                        if progress:
                            progress(result.bytes_done, total, relative)
        except Exception as exc:
            _remove(partial)
            result.failed.append((relative, f"{type(exc).__name__}: {exc}"))
            continue

        # The check that makes everything above safe to have done.
        try:
            actual = hash_file(partial)
        except OSError as exc:
            _remove(partial)
            result.failed.append((relative, f"unreadable after download: {exc}"))
            continue

        if actual != expected:
            _remove(partial)
            result.failed.append((relative, "hash mismatch"))
            continue

        try:
            os.replace(partial, target)
        except OSError as exc:
            _remove(partial)
            result.failed.append((relative, f"could not stage: {exc}"))
            continue

        result.staged.append(relative)

    return result


def _remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def staged_bytes(staging_dir: str) -> int:
    """How much of a part-finished download is currently on disk."""
    total = 0
    for dirpath, _, filenames in os.walk(staging_dir):
        for name in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, name))
            except OSError:
                pass
    return total
