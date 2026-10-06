"""A named lock the app holds while it runs, so Setup and the uninstaller can tell.

Inno Setup's ``AppMutex`` is how an installer asks "is the app open?": Setup and
the uninstaller look for the named mutex and, while it exists, refuse to go on
and ask the user to close the app. Without it, nothing asked. Uninstalling with
the app open deleted what it could and left every file the running app held
(the exe, its DLLs, the packs) behind in a folder Apps & features no longer
lists; installing over it failed partway through the same files.

The name is the installer's ``AppName`` plus " is running" — the Pro build
passes its own AppName, so each edition's installer waits only for its own app.
Both the session-local name and the ``Global\\`` one are created, as Inno's
documentation recommends, so an all-users uninstall started from another
session (an administrator's) still sees an app open in this one.

Held by the process for as long as it lives; Windows releases it when the
process ends, however it ends, so there is nothing to clean up.
"""
from __future__ import annotations

import sys

_handles = []


def app_name(edition: str) -> str:
    """The installer's AppName for this edition (see videohighlighter-packs.iss)."""
    edition = (edition or "").strip()
    if not edition or edition.lower() == "free":
        return "VideoHighlighter"
    return f"VideoHighlighter {edition}"


def mutex_name(edition: str) -> str:
    return f"{app_name(edition)} is running"


def hold(edition: str) -> bool:
    """Create the mutex for this edition. True if one was created; never raises."""
    if sys.platform != "win32" or _handles:
        return bool(_handles)
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateMutexW.restype = wintypes.HANDLE
        kernel32.CreateMutexW.argtypes = (ctypes.c_void_p, wintypes.BOOL,
                                          wintypes.LPCWSTR)
        name = mutex_name(edition)
        for full in (name, "Global\\" + name):
            handle = kernel32.CreateMutexW(None, False, full)
            if handle:
                _handles.append(handle)
            else:
                print(f"app_mutex: could not create {full!r} "
                      f"(error {ctypes.get_last_error()})")
    except Exception as exc:
        # Only the installer reads this; the app runs the same without it.
        print(f"app_mutex: not held ({type(exc).__name__}: {exc})")
    return bool(_handles)
