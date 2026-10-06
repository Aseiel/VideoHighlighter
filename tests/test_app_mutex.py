"""The app holds the mutex the installers wait on, under the same name.

Setup and the uninstaller refuse to run while a mutex named in ``AppMutex``
exists. If the app's name and the installer's drift apart nothing fails — the
installer simply stops noticing the app, and uninstalling it while it is open
goes back to leaving its files behind. So the two are pinned together here.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

from modules.system import app_mutex

REPO = Path(__file__).resolve().parent.parent
INSTALLERS = sorted((REPO / "packaging" / "installer").glob("videohighlighter*.iss"))


def _installer_mutex(text: str, app_name: str) -> str:
    define = re.search(r'^#define AppMutex AppName \+ "([^"]*)"$', text, re.M)
    assert define, "AppMutex is no longer AppName + a suffix"
    return app_name + define.group(1)


@pytest.mark.parametrize("iss", INSTALLERS, ids=lambda p: p.name)
def test_each_installer_waits_for_the_app(iss):
    text = iss.read_text(encoding="utf-8")
    default_name = re.search(r'#define AppName "([^"]+)"', text).group(1)
    assert default_name == app_mutex.app_name("Free")
    assert _installer_mutex(text, default_name) == app_mutex.mutex_name("Free")
    # The Pro build passes /DAppName="VideoHighlighter Pro".
    assert (_installer_mutex(text, "VideoHighlighter Pro")
            == app_mutex.mutex_name("Pro"))
    assert re.search(r"^AppMutex=\{#AppMutex\},Global\\\{#AppMutex\}$", text, re.M)


def test_installers_found():
    assert len(INSTALLERS) >= 2


def test_editions_do_not_share_a_name():
    assert app_mutex.mutex_name("Free") != app_mutex.mutex_name("Pro")


@pytest.mark.skipif(sys.platform != "win32", reason="Windows mutex")
def test_hold_creates_a_mutex_others_can_see():
    import ctypes
    from ctypes import wintypes

    assert app_mutex.hold("Free")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenMutexW.restype = wintypes.HANDLE
    kernel32.OpenMutexW.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR)
    SYNCHRONIZE = 0x00100000
    handle = kernel32.OpenMutexW(SYNCHRONIZE, False, app_mutex.mutex_name("Free"))
    assert handle
    kernel32.CloseHandle(handle)
