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

Where the channel file lives
----------------------------
Two places, both asked; the newest version announced wins, and on a tie the
first one listed here:

1. ``<base_url>/channels/<edition>.json`` on the update host, where
   ``base_url`` comes from the ``manifest.json`` this install shipped with.
   CI stamps it at build time (the ``UPDATE_BASE_URL`` repository variable),
   and ``.github/workflows/publish-update.yaml`` writes the channel file there
   once a release is signed. Nothing about the host is compiled into the code,
   so moving the bucket is a variable change, not a release.
2. The marketing site. ``publish-update.yaml`` writes it too when it has a
   token for the site's repository; otherwise it is committed by hand. Every
   build from before the update host existed reads only this one, and so does
   a run from source, which has no ``manifest.json``. Because the newest
   answer wins, a release announced here for download only is still heard by
   installs whose update host names an older one.

The channel file names one signed release manifest per platform
(``"manifests": {"windows": ...}``). A platform with no entry gets the
download page, which is how macOS stays notify-only: an in-place update would
break the app bundle's signature.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import re
from dataclasses import dataclass
from typing import Callable, Optional

from modules.system.app_paths import user_data_dir
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


def install_dir_writable() -> bool:
    """Can the updater replace files where the app is installed?

    Updating in place renames and writes inside the install folder. An install
    for all users sits under Program Files, which a normal user cannot write
    to, and the updater does not ask for elevation. Offering "Download and
    install" there would fail halfway through, so such an install is offered
    the download page instead. From source this is always True: main.py only
    offers self-install to a frozen build anyway.
    """
    import sys
    if not getattr(sys, "frozen", False):
        return True
    from modules.system.app_paths import _is_writable
    return _is_writable(os.path.dirname(sys.executable))


def _channel() -> str:
    """``"pro"`` or ``"free"`` — which manifest this build should read."""
    return "pro" if (__edition__ or "").strip().lower() == "pro" else "free"


def manifest_url() -> str:
    """The channel file on the marketing site (the fallback; see above)."""
    return f"{_MANIFEST_BASE}/{_channel()}.json"


def update_host() -> str:
    """The update host's base URL from this install's manifest, or ``""``."""
    try:
        from modules.update import update_apply, update_manifest
        installed = update_manifest.load_installed_manifest(
            update_apply.install_root()) or {}
    except Exception:
        return ""
    base = str(installed.get("base_url") or "").strip().rstrip("/")
    return base if base.startswith("https://") else ""


def channel_urls() -> list:
    """Where to look for the channel file, in order."""
    urls = []
    host = update_host()
    if host:
        urls.append(f"{host}/channels/{_channel()}.json")
    urls.append(manifest_url())
    return urls


def _platform_manifest(channel: dict) -> str:
    """The signed release manifest this platform should install, or ``""``."""
    from modules.update.update_manifest import platform_key

    platform = platform_key()
    manifests = channel.get("manifests")
    if isinstance(manifests, dict):
        return str(manifests.get(platform) or "")
    # The single-URL form predates per-platform entries, and every manifest it
    # ever named was a Windows build.
    if platform == "windows":
        return str(channel.get("manifest_url") or "")
    return ""


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

    from modules.system import https_certs

    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            # Identifies the build to the CDN log only; no user, no machine.
            "User-Agent": f"VideoHighlighter/{__version__}",
        },
        method="GET",
    )
    # The CA bundle matters here: a frozen macOS build has no usable platform
    # trust store, and without one this fails as "no manifest" — which reads to
    # the user as "you are up to date". See modules/system/https_certs.py.
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS,
                                **https_certs.opener_kwargs()) as response:
        return json.loads(response.read().decode("utf-8"))


# ---------------------------------------------------------------------------
# The check
# ---------------------------------------------------------------------------

def _self_install_platform() -> bool:
    import sys
    return sys.platform == "win32"


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
        """Offer "Download and install" rather than the download page?

        Only on Windows (a macOS app bundle is signed as a whole, and an
        in-place update breaks the seal), only when this build can verify a
        release at all, and only where the install folder can be written.
        """
        from modules.update import update_manifest

        return (bool(self.manifest_url)
                and _self_install_platform()
                and bool(update_manifest.RELEASE_PUBLIC_KEY_HEX)
                and install_dir_writable())

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
    manifest = None
    try:
        # Every source is asked and the newest answer wins. Stopping at the
        # first one that answers let a stale channel on the update host hide a
        # newer release announced on the site, so an install never heard of it.
        # On a tie the earlier source keeps it: the update host comes first and
        # is the one whose channel names signed manifests.
        for url in channel_urls():
            try:
                answer = fetch(url)
            except Exception as exc:
                # Offline is the common case, not an error worth showing anyone.
                print(f"update_check: no manifest at {url} "
                      f"({type(exc).__name__}: {exc})")
                continue
            if not isinstance(answer, dict):
                print(f"update_check: {url} is not a JSON object; ignoring")
                continue
            if manifest is None or is_newer(str(answer.get("version", "")),
                                            str(manifest.get("version", ""))):
                manifest = answer
        if manifest is None:
            return None
    finally:
        if not force:
            mark_checked()

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
        manifest_url=_platform_manifest(manifest),
    )
