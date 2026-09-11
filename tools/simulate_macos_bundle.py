"""Run the packaged-macOS code paths on a machine that is not a Mac.

The 0.11.0 .app failed three ways at once, and all three were decided by the
*environment* rather than by macOS itself: a working directory of ``/``, a
``sys.executable`` inside a read-only bundle, and no usable certificate store.
Every one of those can be faked here, and then the app's own entry points can
be driven against the fake to see what they decide.

    python -m tools.simulate_macos_bundle            # the whole report
    python -m tools.simulate_macos_bundle --old      # the 0.11.0 behaviour, for contrast

What this can and cannot tell you
---------------------------------
It exercises the plumbing: where the app decides to write, whether a relative
``./cache`` lands somewhere writable, whether the cache the timeline viewer
builds actually gets created, and whether the HTTPS calls carry a certificate
store. Those are the three defects the user hit, and each is invisible from
Windows without this.

It cannot tell you anything about the **bundle**: whether PyInstaller collects
``libtbb.12.dylib``, whether ad-hoc signing holds, whether the .dmg opens. That
is what the macOS runner and a real download settle, and nothing here
substitutes for it.

Dev-only, under `tools/`, and the fake is torn down on the way out.
"""
from __future__ import annotations

import argparse
import contextlib
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS, FAIL, INFO = "✅", "❌", "  "


@contextlib.contextmanager
def _patched(owner, name, value):
    missing = object()
    old = getattr(owner, name, missing)
    setattr(owner, name, value)
    try:
        yield
    finally:
        if old is missing:
            delattr(owner, name)
        else:
            setattr(owner, name, old)


@contextlib.contextmanager
def frozen_macos_app(root):
    """Everything a packaged .app tells Python about itself.

    The working directory stands in for the ``/`` macOS hands an .app. It is a
    directory inside the temp tree rather than the real drive root, for two
    reasons: a simulation must not litter the machine it runs on, and on POSIX
    it can be chmod'ed read-only so the failure reproduces exactly. On Windows
    it stays writable, so there the cache check is judged by *where* the cache
    landed rather than by whether the mkdir raised.
    """
    bundle = os.path.join(root, "VideoHighlighter.app", "Contents", "MacOS")
    os.makedirs(bundle, exist_ok=True)
    home = os.path.join(root, "home")
    os.makedirs(home, exist_ok=True)

    before = os.getcwd()
    with contextlib.ExitStack() as stack:
        stack.enter_context(_patched(sys, "frozen", True))
        stack.enter_context(_patched(sys, "platform", "darwin"))
        stack.enter_context(_patched(sys, "executable",
                                     os.path.join(bundle, "VideoHighlighter")))
        stack.enter_context(_patched(sys, "_MEIPASS", bundle))
        for var in ("HOME", "USERPROFILE"):
            old = os.environ.get(var)
            os.environ[var] = home
            stack.callback(lambda v=var, o=old:
                           os.environ.__setitem__(v, o) if o is not None
                           else os.environ.pop(v, None))
        fake_root = os.path.join(root, "fake-root")
        os.makedirs(fake_root, exist_ok=True)
        os.chdir(fake_root)
        stack.callback(os.chdir, before)
        if os.name != "nt":
            os.chmod(fake_root, 0o555)           # read-only, as "/" is
            stack.callback(os.chmod, fake_root, 0o755)
        yield home


def _old_user_data_dir():
    """What 0.11.0 did: write inside the bundle, whatever the platform."""
    return os.path.dirname(sys.executable)


def check_where_it_writes(old: bool) -> bool:
    from modules import app_paths

    target = _old_user_data_dir() if old else app_paths.user_data_dir()
    inside_bundle = ".app" in target
    print(f"{FAIL if inside_bundle else PASS} user_data_dir() -> {target}")
    if inside_bundle:
        print(f"{INFO}   inside the bundle: read-only under translocation, and "
              f"writing there breaks the signature")
    return not inside_bundle


def check_working_directory(old: bool) -> bool:
    from modules import app_paths

    if old:
        cwd = os.getcwd()                         # 0.11.0 left it alone
    else:
        cwd = app_paths.use_writable_cwd()
    resolved = os.path.abspath("cache")
    ok = os.access(cwd, os.W_OK) and ".app" not in resolved
    print(f"{PASS if ok else FAIL} working directory -> {cwd}")
    print(f"{INFO}   './cache' resolves to {resolved}")
    return ok


def check_the_cache_opens(old: bool) -> bool:
    """The actual failure: VideoAnalysisCache mkdir'ing a relative path.

    This is the call in `modules/video_cache.py` that the timeline viewer and
    the subtitles-on-demand run both die in.
    """
    try:
        from modules.video_cache import VideoAnalysisCache
    except Exception as e:
        print(f"{INFO} cache module unavailable here ({type(e).__name__}: {e})")
        return True
    from modules import app_paths

    try:
        cache = VideoAnalysisCache(cache_dir="./cache")
    except OSError as e:
        # What macOS gives: [Errno 30] Read-only file system: 'cache'.
        print(f"{FAIL} VideoAnalysisCache('./cache') -> {type(e).__name__}: {e}")
        return False

    where = os.path.abspath(str(cache.cache_dir))
    home = os.path.abspath(app_paths.user_data_dir())
    ok = os.path.commonpath([where, home]) == home
    print(f"{PASS if ok else FAIL} VideoAnalysisCache('./cache') created {where}")
    if not ok:
        # On Windows the simulated root is writable, so the mkdir succeeds where
        # macOS would refuse. The location still says the same thing.
        print(f"{INFO}   which is not under {home} — on macOS this is the "
              f"read-only filesystem error")
    return ok


def check_certificates(old: bool) -> bool:
    from modules import https_certs

    if old:
        print(f"{FAIL} HTTPS used urllib's default store "
              f"(macOS: CERTIFICATE_VERIFY_FAILED, reported as 'no manifest')")
        return False
    https_certs.reset_cache()
    kwargs = https_certs.opener_kwargs()
    ok = bool(kwargs)
    print(f"{PASS if ok else FAIL} update calls carry a CA bundle: {list(kwargs) or 'nothing'}")
    if ok:
        try:
            count = len(kwargs["context"].get_ca_certs())
            print(f"{INFO}   {count} certificates loaded from certifi")
        except Exception:
            pass
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--old", action="store_true",
                    help="simulate the 0.11.0 behaviour instead of the fix")
    args = ap.parse_args(argv)

    label = "0.11.0" if args.old else "current"
    print(f"\nPretending to be a frozen macOS .app — {label} behaviour\n")

    results = []
    with tempfile.TemporaryDirectory() as root:
        with frozen_macos_app(root):
            for check in (check_where_it_writes, check_working_directory,
                          check_the_cache_opens, check_certificates):
                results.append(check(args.old))
                print()

    passed = sum(1 for r in results if r)
    print(f"{passed}/{len(results)} checks passed "
          f"({'expected to fail — this is the bug' if args.old else 'the fix'})")
    print("\nThe bundle itself — libtbb, ad-hoc signing, the .dmg — is not "
          "simulated. Only the macOS runner settles that.")
    return 0 if all(results) or args.old else 1


if __name__ == "__main__":
    raise SystemExit(main())
