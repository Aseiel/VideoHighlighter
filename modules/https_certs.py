"""A certificate store the packaged app can actually find.

Python verifies HTTPS against the platform's trust store, and on macOS it does
not read the Keychain: it reads an OpenSSL directory that a stock python.org
build populates with the `Install Certificates.command` script. A frozen app
runs that build without ever running that script, so every HTTPS call fails
with::

    <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed:
    unable to get local issuer certificate>

Observed on the 0.11.0 macOS build: the update check failed at launch and the
app reported "no manifest", which reads as "you are up to date" and is not the
same thing at all.

`certifi` ships Mozilla's CA bundle as a file inside the bundle, which sidesteps
the platform store entirely. It is already present — `requests` depends on it —
and PyInstaller's stock hook collects its `cacert.pem`.

Windows is unaffected (Python reads the system store there), so this is a
fallback rather than a replacement: when certifi is missing, callers get None
and urllib's default behaviour, which is what they had.
"""

from __future__ import annotations

import functools
from typing import Optional


@functools.lru_cache(maxsize=1)
def context():
    """An ``ssl.SSLContext`` built on certifi's bundle, or None.

    Cached: building a context parses the whole CA file, and the update check
    runs at launch where that is measurable.

    None means "let urllib decide", which is correct on a platform whose store
    works and is the only honest answer when there is no bundle to point at.
    """
    try:
        import ssl

        import certifi
    except Exception:  # noqa: BLE001 - no certifi is a normal answer
        return None
    try:
        return ssl.create_default_context(cafile=certifi.where())
    except Exception as e:  # noqa: BLE001 - a broken bundle must not stop a launch
        print(f"⚠️ certifi CA bundle unusable ({type(e).__name__}: {e})")
        return None


def reset_cache():
    """Forget the cached context. For tests."""
    context.cache_clear()


def opener_kwargs() -> dict:
    """``{"context": ...}`` for ``urlopen``, or ``{}``.

    Written as kwargs rather than a positional argument so a call site that
    already passes a timeout stays readable, and so passing nothing at all is
    the natural result when there is nothing to pass.
    """
    ctx: Optional[object] = context()
    return {"context": ctx} if ctx is not None else {}
