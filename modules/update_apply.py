"""Move a verified, staged update into the installed app.

The Windows problem
-------------------
A running program cannot have its own ``.exe`` or its loaded ``.dll`` files
overwritten or deleted — the OS holds them open. Every self-updater has to work
around this. The usual answer is a second helper process that waits for the app
to exit and then swaps the files, which means shipping and maintaining a second
executable.

The trick used here instead: Windows *does* allow a locked file to be
**renamed**. So each file being replaced is first moved aside into a
``.update-old`` folder, which succeeds even while it is loaded, and the new
file is written to the now-free path. The running process keeps using the
renamed copy until it exits — nothing crashes mid-swap — and the leftovers are
swept on the next launch, when nothing holds them any more.

Failure handling
----------------
Every move is journalled, and any failure rolls the whole thing back: files
moved aside go back where they came from, files already placed are removed. A
half-applied update is the one outcome worth real effort to avoid, because it
leaves a program that cannot start and cannot repair itself.

The installed ``manifest.json`` is rewritten **last**, only after every file is
in place. If anything interrupts the process before that, the old manifest
still describes the install honestly, so the next update plan simply resumes.
"""
from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass, field
from typing import Optional

from modules.update_manifest import (
    MANIFEST_FILENAME,
    is_safe_relpath,
    local_path,
)

TRASH_DIRNAME = ".update-old"


@dataclass
class ApplyResult:
    replaced: list = field(default_factory=list)
    removed: list = field(default_factory=list)
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


def trash_dir(root: str) -> str:
    return os.path.join(root, TRASH_DIRNAME)


def _displace(root: str, relative: str, journal: list) -> None:
    """Move an existing file aside so its path can be rewritten.

    Rename rather than delete: the file may be the running executable or a
    loaded DLL, which Windows refuses to remove but allows to be moved.
    """
    current = local_path(root, relative)
    if not os.path.exists(current):
        return
    aside = os.path.join(trash_dir(root), relative.replace("/", os.sep))
    os.makedirs(os.path.dirname(aside) or ".", exist_ok=True)
    if os.path.exists(aside):
        os.remove(aside)
    os.rename(current, aside)
    journal.append((current, aside))


def apply_update(root: str, staging_dir: str, staged_paths: list,
                 delete_paths: Optional[list] = None,
                 manifest_bytes: Optional[bytes] = None) -> ApplyResult:
    """Put the staged files into ``root``. All of it, or none of it.

    ``staged_paths`` are relative paths already downloaded AND hash-verified by
    ``update_download`` — this function does not re-verify, it moves.
    """
    result = ApplyResult()
    journal: list = []      # (original_path, moved_to) — for rollback
    placed: list = []       # paths written, removed on rollback

    try:
        for relative in staged_paths:
            if not is_safe_relpath(relative):
                raise ValueError(f"unsafe path in staged set: {relative!r}")
            source = local_path(staging_dir, relative)
            if not os.path.exists(source):
                raise FileNotFoundError(f"staged file missing: {relative}")

            _displace(root, relative, journal)
            target = local_path(root, relative)
            os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
            shutil.move(source, target)
            placed.append(target)
            result.replaced.append(relative)

        for relative in (delete_paths or []):
            if not is_safe_relpath(relative):
                continue
            _displace(root, relative, journal)
            result.removed.append(relative)

        # Last, so an interruption before this leaves an install whose manifest
        # still tells the truth about it.
        if manifest_bytes is not None:
            with open(os.path.join(root, MANIFEST_FILENAME), "wb") as handle:
                handle.write(manifest_bytes)

    except Exception as exc:
        _rollback(journal, placed)
        result.error = f"{type(exc).__name__}: {exc}"
        result.replaced = []
        result.removed = []

    return result


def _rollback(journal: list, placed: list) -> None:
    """Undo a partial apply: remove what was written, restore what was moved."""
    for path in placed:
        try:
            os.remove(path)
        except OSError:
            pass
    for original, aside in reversed(journal):
        try:
            os.makedirs(os.path.dirname(original) or ".", exist_ok=True)
            os.rename(aside, original)
        except OSError:
            # Nothing better is available here; the file is still in the trash
            # folder, and the next update will re-download it from the manifest.
            print(f"update_apply: could not restore {original}")


def sweep_old(root: str) -> int:
    """Delete the displaced files from a previous update. Call at startup.

    They are only removable once the process that had them open has exited,
    which is why this runs on the next launch rather than at the end of the
    update. Returns the number of bytes reclaimed.
    """
    path = trash_dir(root)
    if not os.path.isdir(path):
        return 0
    freed = 0
    for dirpath, _, filenames in os.walk(path):
        for name in filenames:
            try:
                freed += os.path.getsize(os.path.join(dirpath, name))
            except OSError:
                pass
    shutil.rmtree(path, ignore_errors=True)
    return freed


def install_root() -> str:
    """The directory the update would be applied to."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def relaunch_command() -> list:
    """Argv for restarting the app after an update."""
    if getattr(sys, "frozen", False):
        return [sys.executable]
    return [sys.executable, os.path.join(install_root(), "main.py")]
