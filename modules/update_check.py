"""Tell the user when a newer build exists.

Why this is notify-only
-----------------------
The app is delivered as a multi-gigabyte archive, so "download and run the new
one for you" would mean fetching several GB and then executing it — a remote
code execution channel into every customer machine, and the one bug class that
cannot be recalled once shipped. The value the user actually wants is *knowing*
a new build exists; the download itself is one click on a page. So this module
fetches a small JSON manifest, compares versions, and stops there.

The manifest format already carries ``assets`` (with sizes and SHA-256) and a
``signature`` field, unused today. That is deliberate: if downloading is added
later, the wire format does not change, and the rule for that day is written
down here — **verify the signature against an embedded public key before any
downloaded byte is executed**, exactly as ``licensing.py`` verifies tokens.
GitHub account compromise and MITM both stop at that check; without it, neither
does.

What is sent
------------
Nothing. A plain GET of a public static file, no query string, no identifiers,
no license key. The manifest host learns an IP made a request, which is what
any download link already reveals.

Failure policy
--------------
Every failure is silent — no network, DNS down, garbage JSON, a manifest that
has not been published yet. An update check must never produce an error dialog
or block startup; the user did not ask for it.

Edition
-------
The manifest URL is chosen from ``version.__edition__``, so this file is
identical in the Pro and free repos and picks its own channel at runtime.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import re
from dataclasses import dataclass
from typing import Callable, Optional

from modules.app_paths import user_data_dir
from version import __edition__, __version__

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Static files on the marketing site (GitHub Pages). A release is announced by
# committing one file there — no server, and nothing that can go down
# independently of the site itself.
_MANIFEST_BASE = "https://aseiel.github.io/VideoHighlighter-site/updates"

TIMEOUT_SECONDS = 8
CHECK_INTERVAL_HOURS = 24
STATE_FILENAME = "update_state.json"

# Where the user goes to get the new build. Pro downloads are re-delivered
# through the Lemon Squeezy order portal with the same link they bought with;
# the manifest can override this per release.
_DEFAULT_LANDING = {
    "pro": "https://app.lemonsqueezy.com/my-orders",
    "free": "https://github.com/Aseiel/VideoHighlighter/releases/latest",
}


def _channel() -> str:
    """``"pro"`` or ``"free"`` — which manifest this build should read."""
    return "pro" if (__edition__ or "").strip().lower() == "pro" else "free"


def manifest_url() -> str:
    return f"{_MANIFEST_BASE}/{_channel()}.json"


# ---------------------------------------------------------------------------
# Version comparison
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(r"\d+")


def parse_version(text: str) -> tuple:
    """``"0.9.1"`` -> ``(0, 9, 1)``; tolerant of junk and suffixes.

    Deliberately lenient rather than strict: a malformed version in a manifest
    must not raise on a user's machine. Anything unparseable becomes ``()``,
    which compares lower than every real version, so garbage never announces an
    update.
    """
    return tuple(int(n) for n in _NUM_RE.findall(str(text or "")))


def is_newer(candidate: str, current: str) -> bool:
    """Whether ``candidate`` is a strictly newer version than ``current``."""
    a, b = parse_version(candidate), parse_version(current)
    if not a:
        return False
    # Pad so (0, 9) and (0, 9, 0) compare equal rather than by length.
    width = max(len(a), len(b))
    return a + (0,) * (width - len(a)) > b + (0,) * (width - len(b))


# ---------------------------------------------------------------------------
# Local state (throttle + "skip this version")
# ---------------------------------------------------------------------------

def state_path() -> str:
    return os.path.join(user_data_dir(), STATE_FILENAME)


def load_state() -> dict:
    try:
        with open(state_path(), "r", encoding="utf-8") as handle:
            state = json.load(handle)
        return state if isinstance(state, dict) else {}
    except (OSError, ValueError):
        return {}


def save_state(state: dict) -> None:
    try:
        with open(state_path(), "w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2)
    except OSError as exc:
        print(f"update_check: could not save state: {exc}")


def is_enabled() -> bool:
    """False once the user has switched update checks off."""
    return bool(load_state().get("enabled", True))


def set_enabled(enabled: bool) -> None:
    state = load_state()
    state["enabled"] = bool(enabled)
    save_state(state)


def skip_version(version: str) -> None:
    """Silence notifications for ``version`` — until a newer one appears."""
    state = load_state()
    state["skipped_version"] = str(version)
    save_state(state)


def mark_checked(now: Optional[_dt.datetime] = None) -> None:
    state = load_state()
    now = now or _dt.datetime.now(_dt.timezone.utc)
    state["last_check"] = now.isoformat()
    save_state(state)


def due_for_check(now: Optional[_dt.datetime] = None) -> bool:
    """Whether enough time has passed since the last check.

    Startup is the only moment this runs, so without a throttle a user who
    opens the app ten times a day makes ten requests to learn the same thing.
    """
    state = load_state()
    if not state.get("enabled", True):
        return False
    last = state.get("last_check")
    if not last:
        return True
    try:
        when = _dt.datetime.fromisoformat(str(last).replace("Z", "+00:00"))
    except ValueError:
        return True
    now = now or _dt.datetime.now(_dt.timezone.utc)
    if when.tzinfo is None:
        when = when.replace(tzinfo=_dt.timezone.utc)
    return (now - when) >= _dt.timedelta(hours=CHECK_INTERVAL_HOURS)


# ---------------------------------------------------------------------------
# Transport (stdlib only; injectable for tests)
# ---------------------------------------------------------------------------

def _get_json(url: str) -> dict:
    """GET ``url`` and parse it as JSON.

    ``urllib`` rather than ``requests``/``httpx``: this runs at startup in a
    frozen exe, and the licensing modules already set the precedent that
    security-adjacent network code carries no new dependency.
    """
    import urllib.request

    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            # Identifies the build to the CDN log only; no user, no machine.
            "User-Agent": f"VideoHighlighter/{__version__}",
        },
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        return json.loads(response.read().decode("utf-8"))


# ---------------------------------------------------------------------------
# The check
# ---------------------------------------------------------------------------

@dataclass
class UpdateInfo:
    """A newer release the user has not already skipped."""

    version: str
    date: str = ""
    notes: str = ""
    notes_url: str = ""
    download_url: str = ""
    # Where the signed per-file release manifest lives. Empty means this
    # release can only be installed by hand — the banner then offers the
    # download page instead of installing anything, which is also the correct
    # behaviour for every build published before the updater existed.
    manifest_url: str = ""

    @property
    def can_self_install(self) -> bool:
        return bool(self.manifest_url)

    @property
    def headline(self) -> str:
        return f"Version {self.version} is available (you have {__version__})."


def check_for_update(
    *,
    current_version: Optional[str] = None,
    force: bool = False,
    transport: Optional[Callable[[str], dict]] = None,
) -> Optional[UpdateInfo]:
    """The newer release, or ``None``.

    Blocking — call it off the GUI thread. ``None`` covers every uninteresting
    outcome: up to date, throttled, disabled, offline, malformed manifest, or a
    version the user chose to skip. Callers show a notification if and only if
    this returns something.

    ``force=True`` bypasses the throttle and the skip list, for a "Check for
    updates" menu item where the user is explicitly asking.
    """
    if not force and not due_for_check():
        return None

    current = current_version or __version__
    fetch = transport or _get_json
    try:
        manifest = fetch(manifest_url())
    except Exception as exc:
        # Offline is the common case, not an error worth showing anyone.
        print(f"update_check: no manifest ({type(exc).__name__}: {exc})")
        return None
    finally:
        if not force:
            mark_checked()

    if not isinstance(manifest, dict):
        print("update_check: manifest is not a JSON object; ignoring")
        return None

    latest = str(manifest.get("version", "")).strip()
    if not is_newer(latest, current):
        return None

    if not force:
        skipped = str(load_state().get("skipped_version", ""))
        # Only the exact skipped version stays silent: skipping 0.9.1 must not
        # also swallow 0.9.2.
        if skipped and not is_newer(latest, skipped):
            return None

    return UpdateInfo(
        version=latest,
        date=str(manifest.get("date", "")),
        notes=str(manifest.get("notes", "")),
        notes_url=str(manifest.get("notes_url", "")),
        download_url=str(
            manifest.get("download_url") or _DEFAULT_LANDING[_channel()]
        ),
        manifest_url=str(manifest.get("manifest_url") or ""),
    )
