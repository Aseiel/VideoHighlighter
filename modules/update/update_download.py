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
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Optional

from modules.update.update_manifest import hash_file, is_safe_relpath, local_path

TIMEOUT_SECONDS = 30
WORKERS = 6
# Waits before the second and third attempt at a file. A dropped connection
# halfway through a multi-GB update should cost one file a few seconds, not
# the user a "try again".
RETRY_DELAYS = (2.0, 6.0)
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

    from modules.system import https_certs

    request = urllib.request.Request(url, headers=headers or {}, method="GET")
    return urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS,
                                  **https_certs.opener_kwargs())


# Blob encodings a signed manifest may declare. ``gzip`` blobs live at
# ``files/<sha256>.gz``; the hash is always of the *decompressed* file, so
# compression changes what travels and nothing about what is verified.
COMPRESSIONS = ("", "gzip")


def file_url(base_url: str, entry: dict, layout: str = "content",
             compression: str = "") -> str:
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
    suffix = ".gz" if compression == "gzip" else ""
    if layout == "path":
        return base + "/" + quote(entry["path"]) + suffix
    return base + "/files/" + quote(str(entry["sha256"])) + suffix


def _is_permanent(exc: Exception) -> bool:
    """Whether retrying ``exc`` is pointless.

    A 4xx means the host answered and the blob is not there (or not ours to
    have); asking again returns the same. Everything else — a reset, a timeout,
    a 5xx from a CDN edge — is the kind of thing a second attempt fixes.
    """
    code = getattr(exc, "code", None)
    return isinstance(code, int) and 400 <= code < 500


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
    workers: int = WORKERS,
    compression: str = "",
) -> DownloadResult:
    """Download every file in ``plan`` into ``staging_dir``.

    ``progress(bytes_done, bytes_total, current_path)`` is called as data
    arrives — often enough for a progress bar, not per chunk of every file.
    ``should_cancel()`` is polled between chunks so a user can stop a multi-GB
    download without waiting for it to finish.

    Files are fetched ``workers`` at a time. An update is mostly small files —
    Python modules, Qt plugins — and one at a time the round trips, not the
    bytes, set the pace. Callbacks are still made one at a time, from whichever
    worker has news, so a caller never sees two at once.
    """
    if compression not in COMPRESSIONS:
        raise ValueError(f"unsupported blob compression {compression!r}")
    result = DownloadResult()
    total = plan.download_bytes
    fetch = opener or _default_opener
    lock = threading.Lock()
    stop = threading.Event()

    def cancelled() -> bool:
        if stop.is_set():
            return True
        if should_cancel:
            with lock:
                if should_cancel():
                    stop.set()
        return stop.is_set()

    def advance(count: int, relative: str) -> None:
        with lock:
            result.bytes_done += count
            if progress:
                progress(result.bytes_done, total, relative)

    def fail(relative: str, reason: str) -> None:
        with lock:
            result.failed.append((relative, reason))

    def one(entry: dict) -> None:
        relative = entry.get("path")
        if not is_safe_relpath(relative):
            fail(str(relative), "unsafe path")
            return
        if cancelled():
            return

        target = local_path(staging_dir, relative)
        expected = entry.get("sha256")

        # Already staged from an earlier attempt?
        if os.path.exists(target):
            try:
                if hash_file(target) == expected:
                    with lock:
                        result.staged.append(relative)
                    advance(int(entry.get("size", 0)), relative)
                    return
            except OSError:
                pass
            _remove(target)

        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        partial = target + ".part"
        url = file_url(base_url, entry, layout, compression)

        for attempt in range(len(RETRY_DELAYS) + 1):
            written = 0
            # Progress counts bytes of the file as it will be on disk, so the
            # bar's total (the manifest's sizes) means the same either way.
            inflate = (zlib.decompressobj(16 + zlib.MAX_WBITS)
                       if compression == "gzip" else None)
            try:
                with fetch(url, headers or {}) as response:
                    with open(partial, "wb") as handle:
                        while True:
                            if cancelled():
                                break
                            block = response.read(_CHUNK)
                            if not block:
                                if inflate is not None:
                                    block = inflate.flush()
                                    if not inflate.eof:
                                        raise EOFError("compressed blob ended early")
                                    inflate = None
                                    if block:
                                        handle.write(block)
                                        written += len(block)
                                        advance(len(block), relative)
                                break
                            if inflate is not None:
                                block = inflate.decompress(block)
                            handle.write(block)
                            written += len(block)
                            advance(len(block), relative)
                if cancelled():
                    _remove(partial)
                    advance(-written, relative)
                    return
                break
            except Exception as exc:
                _remove(partial)
                # Take back what this attempt counted, so a retry does not
                # push the progress bar past 100%.
                advance(-written, relative)
                if attempt >= len(RETRY_DELAYS) or _is_permanent(exc) or cancelled():
                    fail(relative, f"{type(exc).__name__}: {exc}")
                    return
                print(f"update_download: {relative}: {type(exc).__name__}: "
                      f"{exc}; retrying")
                time.sleep(RETRY_DELAYS[attempt])

        # The check that makes everything above safe to have done.
        try:
            actual = hash_file(partial)
        except OSError as exc:
            _remove(partial)
            fail(relative, f"unreadable after download: {exc}")
            return

        if actual != expected:
            _remove(partial)
            fail(relative, "hash mismatch")
            return

        try:
            os.replace(partial, target)
        except OSError as exc:
            _remove(partial)
            fail(relative, f"could not stage: {exc}")
            return

        with lock:
            result.staged.append(relative)

    entries = list(plan.download)
    if workers <= 1 or len(entries) <= 1:
        for entry in entries:
            one(entry)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for future in [pool.submit(one, entry) for entry in entries]:
                future.result()

    result.cancelled = stop.is_set()
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
