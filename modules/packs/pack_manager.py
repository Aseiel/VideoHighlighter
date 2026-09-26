"""Download and install the optional packs from inside the app, on first use.

The installer (packaging/installer/videohighlighter-packs.iss) offers the
NVIDIA PyTorch pack and the CLIP model as checkboxes during setup. The portable
download has no setup step, so the app itself fetches a pack the first time a
feature needs it: the user picks the NVIDIA backend, or opens visual search.
This module does that for both kinds of install, and holds no Qt, so it is
testable without a window or a network (``pack_ui`` is the Qt layer).

**Which packs, and what they must hash to, is fixed at build time.** CI writes
``packs.lock.json`` beside the exe (tools/write_packs_lock.py): the published
``packs.json`` plus the release URL it came from. Nothing is looked up at
runtime, so an app build always fetches exactly the bytes it was tested with,
and a new packs release cannot change what an old build installs. Moving to
new packs is a new app build with a new lock.

**A pack is not trusted until its SHA-256 matches** the lock. A truncated
download, a proxy's error page served as 200, a swapped asset: all fail the
same check, and nothing reaches ``packs/`` or ``models/`` before it passes.

**Interrupted downloads resume.** The NVIDIA pack is ~2 GB, and a portable
user is exactly the person on a flaky connection. Bytes on disk are kept in
``<asset>.part`` across a cancel, a network error or a restart of the app, and
the next attempt asks for the rest with an HTTP Range request. GitHub's asset
host honours Range; a server that ignores it gets a clean restart, not a file
with the start written twice.

**A PyTorch pack takes effect on the next start.** ``pipeline`` imports torch
when the app starts, from whichever pack ``rthook_packs`` chose, and torch
cannot be swapped inside a running process. So a new pack is placed where the
loader will prefer it next time, and the app asks for a restart:

* a pack that is not installed yet goes straight to ``packs/<name>``; the
  loader already prefers it by ``priority`` (cu128 10, cpu 0);
* one that is (a reinstall or a newer build) goes to ``packs/<name>.pending``,
  because the running process has its DLLs loaded; the loader swaps it in at
  the next start, before anything imports torch;
* the pack it supersedes (torch-cpu) is marked with a ``.remove`` file rather
  than deleted, for the same reason, and the loader clears it next start.

The CLIP model is plain files the app opens when visual search starts, so it
goes straight into ``models/`` and works without a restart.

**Where.** Beside the exe when that folder is writable, which keeps a portable
install portable; the per-user folder otherwise (``app_paths.user_data_dir``,
the same rule the rest of the app's writes follow). The loader and
clip_prefilter look in both.
"""
from __future__ import annotations

import http.client
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Callable, Optional

from modules.update.update_manifest import hash_file

LOCK_NAME = "packs.lock.json"
LOCK_ENV = "VH_PACKS_LOCK"          # a lock file elsewhere, for testing a build
PACKS_DIRNAME = "packs"
MODELS_DIRNAME = "models"
STAGING_DIRNAME = ".pack-staging"
PENDING_SUFFIX = ".pending"
REMOVE_MARKER = ".remove"
APP_DIRNAME = "VideoHighlighter"

# Where each model pack unpacks under models/, and the file that proves it is
# there. The archive carries the folder itself; this is only so presence can be
# checked without it. Same name as ClipDirName in videohighlighter-packs.iss
# and BUNDLED_OV_DIRNAME in llm/clip_prefilter.py.
MODEL_DIRS = {
    "models-clip": ("clip-vit-base-patch32-ov", "openvino_model.xml"),
}

# The NVIDIA pack's names, so callers need not spell them.
CUDA_PACK = "torch-cu128"
CPU_PACK = "torch-cpu"
CLIP_PACK = "models-clip"

# CUDA 12.x minor-version compatibility floor on Windows. PyTorch's cu128
# wheels carry their own CUDA runtime, so what matters is the driver. A card
# new enough to need 12.8 (RTX 50) never has an older driver anyway. Check
# NVIDIA's CUDA compatibility table when the pack moves to a new CUDA major.
MIN_DRIVER = (527, 41)

TIMEOUT_SECONDS = 30
RETRIES = 4                          # per download, on network errors
_CHUNK = 1024 * 1024
_PROGRESS_STEP = 4 * 1024 * 1024     # report every 4 MB, not every chunk
_SPACE_MARGIN = 256 * 1024 * 1024

# Progress phases, as update_install has them, so one progress bar serves both.
DOWNLOADING = "downloading"
VERIFYING = "verifying"
INSTALLING = "installing"

# status()
INSTALLED = "installed"
PENDING = "pending"                  # installed, takes effect on restart
OUTDATED = "outdated"                # another version of the pack is present
MISSING = "missing"


class PackError(Exception):
    """A pack could not be installed; the message is for the user."""


class _Restart(Exception):
    """The partial file cannot be continued; start the download over."""


@dataclass(frozen=True)
class Pack:
    name: str
    dep: str
    version: str
    python: Optional[str]
    asset: str
    bytes: int
    sha256: str
    bytes_installed: int
    url: str

    @property
    def is_model(self) -> bool:
        return self.name in MODEL_DIRS or self.name.startswith("models-")

    @property
    def download_mb(self) -> int:
        return -(-self.bytes // (1024 * 1024))

    @property
    def installed_mb(self) -> int:
        return -(-self.bytes_installed // (1024 * 1024))


@dataclass
class PackResult:
    ok: bool = False
    message: str = ""
    name: str = ""
    restart_required: bool = False
    cancelled: bool = False


ProgressFn = Callable[[str, int, int, str], None]


# ---------------------------------------------------------------------------
# Where things live
# ---------------------------------------------------------------------------

def exe_dir() -> str:
    """The folder of the exe; the project root when running from source."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    # modules/packs/pack_manager.py -> up three to the project root.
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def per_user_dir() -> str:
    """The per-user folder, *not* created. Same rule as app_paths._per_user_dir
    and rthook_packs._roots, which must agree with it."""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser(r"~\AppData\Local")
    elif sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Application Support")
    else:
        base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return os.path.join(base, APP_DIRNAME)


def install_roots() -> list:
    """Every root a pack may be installed under, in the order the loader
    searches them: beside the exe first, then the per-user folder."""
    out, seen = [], set()
    for root in (exe_dir(), per_user_dir()):
        key = os.path.normcase(os.path.abspath(root))
        if key not in seen:
            seen.add(key)
            out.append(root)
    return out


def target_root() -> str:
    """Where a new pack goes: beside the exe when writable, else per-user."""
    from modules.system import app_paths
    return app_paths.user_data_dir()


# ---------------------------------------------------------------------------
# The lock
# ---------------------------------------------------------------------------

def lock_path() -> str:
    return os.environ.get(LOCK_ENV) or os.path.join(exe_dir(), LOCK_NAME)


def load_lock(path: Optional[str] = None) -> dict:
    """``{name: Pack}`` this build may install; empty when it has no lock
    (running from source, or a build that predates packs)."""
    path = path or lock_path()
    try:
        with open(path, encoding="utf-8-sig") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        print(f"[packs] unreadable {path}: {exc}")
        return {}
    base = str(data.get("base") or "").rstrip("/")
    out = {}
    for row in data.get("packs") or []:
        try:
            url = row.get("url") or (base + "/" + row["asset"] if base else "")
            if not url or not row.get("sha256"):
                continue
            pack = Pack(
                name=str(row["name"]), dep=str(row.get("dep") or row["name"]),
                version=str(row.get("version") or ""), python=row.get("python"),
                asset=str(row["asset"]), bytes=int(row["bytes"]),
                sha256=str(row["sha256"]).lower(),
                bytes_installed=int(float(row.get("bytes_installed") or 0)),
                url=url)
        except (KeyError, TypeError, ValueError) as exc:
            print(f"[packs] skipping a malformed lock entry ({exc}): {row!r}")
            continue
        out[pack.name] = pack
    return out


# ---------------------------------------------------------------------------
# What is installed
# ---------------------------------------------------------------------------

def _read_json(path: str) -> dict:
    try:
        with open(path, encoding="utf-8-sig") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _priority(pack_dir: str) -> int:
    try:
        return int(_read_json(os.path.join(pack_dir, "pack.json")).get("priority", 0))
    except (TypeError, ValueError):
        return 0


def _model_roots(roots: list) -> list:
    """Model packs are also found inside the frozen bundle (older builds and
    the macOS bundle carry CLIP there)."""
    meipass = getattr(sys, "_MEIPASS", None)
    return list(roots) + ([meipass] if meipass else [])


def status(pack: Pack, roots: Optional[list] = None) -> str:
    roots = roots or install_roots()
    if pack.is_model:
        dirname, probe = MODEL_DIRS.get(pack.name, (None, None))
        if dirname is None:
            return MISSING
        for root in _model_roots(roots):
            if os.path.isfile(os.path.join(root, MODELS_DIRNAME, dirname, probe)):
                return INSTALLED
        return MISSING

    found = MISSING
    for root in roots:
        target = os.path.join(root, PACKS_DIRNAME, pack.name)
        if os.path.isfile(os.path.join(target + PENDING_SUFFIX, "pack.json")):
            return PENDING
        if os.path.isdir(os.path.join(target, "site-packages")):
            version = _read_json(os.path.join(target, "pack.json")).get("version")
            if not version or version == pack.version:
                return INSTALLED
            found = OUTDATED
    return found


def is_ready(name: str, lock: Optional[dict] = None) -> bool:
    """Installed or waiting for a restart: nothing to download."""
    lock = load_lock() if lock is None else lock
    pack = lock.get(name)
    return pack is not None and status(pack) in (INSTALLED, PENDING)


def python_tag() -> str:
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def incompatibility(pack: Pack) -> Optional[str]:
    """Why this pack cannot work in this app, or None."""
    if pack.python and pack.python != python_tag():
        return (f"The {pack.name} pack is built for Python {pack.python}, but this "
                f"app runs {python_tag()}. The app and its packs are out of step; "
                f"download the current release.")
    return None


# ---------------------------------------------------------------------------
# NVIDIA
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NvidiaGpu:
    name: str
    driver: str


def _no_window() -> dict:
    # The packaged exe is --windowed: without this, every child flashes a console.
    if sys.platform == "win32":
        return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}
    return {}


def _driver_tuple(text: str) -> tuple:
    parts = []
    for piece in str(text).strip().split("."):
        try:
            parts.append(int(piece))
        except ValueError:
            break
    return tuple(parts)


def probe_nvidia(run: Callable = subprocess.run) -> list:
    """NVIDIA cards and their driver, from nvidia-smi (installed with every
    driver). Empty when there is none, or nvidia-smi cannot say. Never imports
    torch, so it is safe to ask before the NVIDIA pack exists."""
    exe = shutil.which("nvidia-smi")
    if not exe and sys.platform == "win32":
        exe = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"),
                           "System32", "nvidia-smi.exe")
    if not exe:
        return []
    try:
        proc = run([exe, "--query-gpu=name,driver_version", "--format=csv,noheader"],
                   capture_output=True, text=True, timeout=15, **_no_window())
    except (OSError, subprocess.SubprocessError):
        return []
    if proc.returncode != 0:
        return []
    gpus = []
    for line in (proc.stdout or "").splitlines():
        name, _, driver = line.rpartition(",")
        if name.strip() and driver.strip():
            gpus.append(NvidiaGpu(name.strip(), driver.strip()))
    return gpus


def cuda_pack_advice(gpus: list) -> Optional[str]:
    """Why downloading the NVIDIA pack would not help on this machine, or None.

    Asked before offering ~2 GB: a user without a usable card should hear so
    now, not after the download. cuda_check still has the last word once torch
    runs (a card newer than the pack's kernels passes this and fails there).
    """
    if not gpus:
        return ("No NVIDIA graphics card was found, so the NVIDIA download would "
                "not speed anything up. VideoHighlighter keeps using the "
                "processor or your Intel/AMD graphics.")
    best = max(gpus, key=lambda g: _driver_tuple(g.driver))
    if _driver_tuple(best.driver) < MIN_DRIVER:
        need = ".".join(str(p) for p in MIN_DRIVER)
        return (f"Your NVIDIA driver ({best.driver}) is too old for GPU "
                f"acceleration. Update it to {need} or newer from nvidia.com, "
                f"then try again.")
    return None


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def _default_opener(url: str, headers: dict):
    import urllib.request

    from modules.system import https_certs

    request = urllib.request.Request(url, method="GET", headers={
        "Accept": "application/octet-stream",
        "User-Agent": "VideoHighlighter-packs",
        **(headers or {}),
    })
    # urllib keeps the Range header across GitHub's redirect to its asset host.
    return urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS,
                                  **https_certs.opener_kwargs())


def _size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def _remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def _http_code(exc) -> Optional[int]:
    code = getattr(exc, "code", None)
    return code if isinstance(code, int) else None


def _fetch_into(fetch, pack: Pack, part: str, progress: Optional[ProgressFn],
                should_cancel: Optional[Callable[[], bool]]) -> bool:
    """Append the rest of ``pack`` to ``part``. False when cancelled."""
    offset = _size(part)
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    try:
        response = fetch(pack.url, headers)
    except Exception as exc:
        if offset and _http_code(exc) == 416:
            raise _Restart() from exc     # the host does not accept our offset
        raise
    with response:
        code = getattr(response, "status", None) or response.getcode()
        if offset:
            if code != 206:
                offset = 0                # Range ignored: the body is the whole file
            else:
                got = (getattr(response, "headers", None) or {}).get("Content-Range", "")
                if got and not got.startswith(f"bytes {offset}-"):
                    raise _Restart()
        done = offset
        reported = done
        with open(part, "ab" if offset else "wb") as fh:
            while True:
                if should_cancel and should_cancel():
                    return False
                block = response.read(_CHUNK)
                if not block:
                    break
                fh.write(block)
                done += len(block)
                if done > pack.bytes:
                    raise _Restart()
                if progress and done - reported >= _PROGRESS_STEP:
                    reported = done
                    progress(DOWNLOADING, done, pack.bytes, pack.asset)
    if progress:
        progress(DOWNLOADING, done, pack.bytes, pack.asset)
    return True


def download_pack(
    pack: Pack,
    dest_dir: str,
    *,
    progress: Optional[ProgressFn] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
    opener: Optional[Callable] = None,
    sleep: Callable[[float], None] = time.sleep,
) -> Optional[str]:
    """Fetch ``pack``'s archive into ``dest_dir`` and verify it.

    Returns the verified file's path, or None when cancelled; the bytes so far
    stay in ``<asset>.part`` for the next attempt. Raises PackError when it
    cannot be done: after RETRIES network failures in a row, at once for a
    404 (a lock pointing at a tag that does not exist), or on a hash mismatch.
    """
    os.makedirs(dest_dir, exist_ok=True)
    final = os.path.join(dest_dir, pack.asset)
    part = final + ".part"
    fetch = opener or _default_opener

    if os.path.isfile(final):
        if _size(final) == pack.bytes and hash_file(final) == pack.sha256:
            return final
        _remove(final)

    failures = 0
    while True:
        if should_cancel and should_cancel():
            return None
        have = _size(part)
        if have > pack.bytes:
            _remove(part)
            have = 0
        if have == pack.bytes:
            break
        try:
            if not _fetch_into(fetch, pack, part, progress, should_cancel):
                return None
            if _size(part) == pack.bytes:
                break
            raise ConnectionError(
                f"connection closed at {_size(part)} of {pack.bytes} bytes")
        except _Restart:
            _remove(part)
            failures += 1
        except (OSError, http.client.HTTPException) as exc:
            code = _http_code(exc)
            if code is not None and 400 <= code < 500 and code not in (408, 416, 429):
                raise PackError(f"{pack.asset} could not be downloaded "
                                f"(HTTP {code} from {pack.url}).") from exc
            failures += 1
            if failures > RETRIES:
                raise PackError(f"{pack.asset} could not be downloaded: {exc}. "
                                f"What arrived is kept; trying again continues "
                                f"from there.") from exc
            print(f"[packs] {pack.asset}: {exc}; retrying ({failures}/{RETRIES})")
        if failures > RETRIES:
            raise PackError(f"{pack.asset}: the server kept sending a different file.")
        sleep(min(2 ** failures, 30))

    if progress:
        progress(VERIFYING, pack.bytes, pack.bytes, pack.asset)
    actual = hash_file(part)
    if actual != pack.sha256:
        _remove(part)
        raise PackError(f"{pack.asset} did not match its checksum and was "
                        f"discarded. Try again; if it keeps happening, the "
                        f"download is being altered on the way.")
    os.replace(part, final)
    return final


# ---------------------------------------------------------------------------
# Unpack
# ---------------------------------------------------------------------------

def find_7z() -> Optional[str]:
    """7-Zip to unpack with: the 7zr.exe shipped beside the app, else one on PATH."""
    names = ("7zr.exe", "7za.exe", "7z.exe") if sys.platform == "win32" else ("7zr", "7za", "7z")
    dirs = [exe_dir()]
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        dirs.append(meipass)
    for folder in dirs:
        for name in names:
            path = os.path.join(folder, name)
            if os.path.isfile(path):
                return path
    for name in names:
        path = shutil.which(name)
        if path:
            return path
    return None


def extract(archive: str, dest: str, *, run: Callable = subprocess.run,
            seven_zip: Optional[str] = None) -> None:
    """Unpack ``archive`` into a fresh ``dest``. Raises PackError."""
    shutil.rmtree(dest, ignore_errors=True)
    os.makedirs(dest, exist_ok=True)
    exe = seven_zip or find_7z()
    if exe:
        proc = run([exe, "x", "-y", "-bso0", "-bsp0", archive, f"-o{dest}"],
                   capture_output=True, text=True, **_no_window())
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
            raise PackError(f"7-Zip could not unpack {os.path.basename(archive)} "
                            f"(exit code {proc.returncode}). {' '.join(tail)}".strip())
        return
    try:
        import py7zr  # dev fallback only; the packaged app ships 7zr.exe
    except ImportError:
        raise PackError("7-Zip was not found, so the download cannot be unpacked. "
                        "7zr.exe belongs next to VideoHighlighter.exe; "
                        "re-extract the portable archive.") from None
    with py7zr.SevenZipFile(archive, "r") as z:
        z.extractall(dest)


# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------

def _free_bytes(path: str) -> int:
    probe = path
    while probe and not os.path.exists(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    try:
        return shutil.disk_usage(probe).free
    except OSError:
        return -1


def _place_python(pack: Pack, unpacked: str, root: str) -> str:
    """Move an unpacked PyTorch pack into ``root/packs``; returns where it went."""
    if not (os.path.isdir(os.path.join(unpacked, "site-packages"))
            and os.path.isfile(os.path.join(unpacked, "pack.json"))):
        raise PackError(f"{pack.asset} does not hold a pack (no site-packages "
                        f"and pack.json at its top).")
    packs = os.path.join(root, PACKS_DIRNAME)
    os.makedirs(packs, exist_ok=True)
    target = os.path.join(packs, pack.name)
    if os.path.exists(target):
        # In use by this process: the loader swaps it in at the next start.
        dest = target + PENDING_SUFFIX
        shutil.rmtree(dest, ignore_errors=True)
    else:
        dest = target
    os.replace(unpacked, dest)
    _mark_superseded(pack, _priority(dest))
    return dest


def _mark_superseded(pack: Pack, priority: int) -> None:
    """Flag every other pack of the same dep with a lower priority, in any root,
    for the loader to clear on the next start (their DLLs are loaded now)."""
    for root in install_roots():
        packs = os.path.join(root, PACKS_DIRNAME)
        try:
            entries = os.listdir(packs)
        except OSError:
            continue
        for entry in entries:
            if entry == pack.name or entry.startswith(".") or "." in entry:
                continue
            folder = os.path.join(packs, entry)
            info = _read_json(os.path.join(folder, "pack.json"))
            if str(info.get("dep") or entry) != pack.dep or _priority(folder) >= priority:
                continue
            try:
                with open(os.path.join(folder, REMOVE_MARKER), "w", encoding="utf-8") as fh:
                    fh.write(pack.name)
            except OSError as exc:
                print(f"[packs] could not mark {folder} for removal: {exc}")


def _place_model(pack: Pack, unpacked: str, root: str) -> str:
    """Move an unpacked model pack's ``models/<dir>`` into ``root/models``."""
    source_models = os.path.join(unpacked, MODELS_DIRNAME)
    dirname = MODEL_DIRS.get(pack.name, (None, None))[0]
    if dirname is None:
        try:
            entries = [e for e in os.listdir(source_models)
                       if os.path.isdir(os.path.join(source_models, e))]
        except OSError:
            entries = []
        dirname = entries[0] if len(entries) == 1 else None
    source = os.path.join(source_models, dirname) if dirname else ""
    if not dirname or not os.path.isdir(source):
        raise PackError(f"{pack.asset} does not hold a models/ folder.")
    models = os.path.join(root, MODELS_DIRNAME)
    os.makedirs(models, exist_ok=True)
    target = os.path.join(models, dirname)
    if os.path.exists(target):
        aside = os.path.join(models, f".{dirname}.old-{os.getpid()}")
        try:
            os.rename(target, aside)
        except OSError as exc:
            raise PackError("The visual search model is in use. Close visual "
                            "search and try again.") from exc
        shutil.rmtree(aside, ignore_errors=True)
    os.replace(source, target)
    return target


def install_pack(
    name: str,
    *,
    lock: Optional[dict] = None,
    root: Optional[str] = None,
    progress: Optional[ProgressFn] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
    opener: Optional[Callable] = None,
    extractor: Optional[Callable[[str, str], None]] = None,
    sleep: Callable[[float], None] = time.sleep,
) -> PackResult:
    """Download, verify, unpack and place one pack. Never raises."""
    lock = load_lock() if lock is None else lock
    pack = lock.get(name)
    if pack is None:
        return PackResult(False, f"This build does not offer {name} "
                                 f"(no {LOCK_NAME} entry).", name)
    reason = incompatibility(pack)
    if reason:
        return PackResult(False, reason, name)
    if status(pack) in (INSTALLED, PENDING):
        return PackResult(True, f"{name} is already installed.", name,
                          restart_required=status(pack) == PENDING)

    root = root or target_root()
    staging = os.path.join(root, STAGING_DIRNAME)
    part = os.path.join(staging, pack.asset + ".part")
    need = pack.bytes - _size(part) + pack.bytes_installed + _SPACE_MARGIN
    free = _free_bytes(root)
    if 0 <= free < need:
        return PackResult(False, (
            f"Not enough free space for {name} on {os.path.splitdrive(os.path.abspath(root))[0] or root}: "
            f"needs about {-(-need // 2**20)} MB, {free // 2**20} MB free."), name)

    unpacked = os.path.join(staging, pack.name + ".unpacked")
    try:
        archive = download_pack(pack, staging, progress=progress,
                                should_cancel=should_cancel, opener=opener, sleep=sleep)
        if archive is None:
            return PackResult(False, "Download paused. It continues where it "
                                     "stopped the next time.", name, cancelled=True)
        if progress:
            progress(INSTALLING, 0, 0, pack.name)
        (extractor or extract)(archive, unpacked)
        if pack.is_model:
            where = _place_model(pack, unpacked, root)
            restart = False
        else:
            where = _place_python(pack, unpacked, root)
            restart = True
    except PackError as exc:
        return PackResult(False, str(exc), name)
    except OSError as exc:
        return PackResult(False, f"{name} could not be installed: {exc}", name)
    finally:
        shutil.rmtree(unpacked, ignore_errors=True)

    _remove(archive)
    try:
        os.rmdir(staging)
    except OSError:
        pass
    print(f"[packs] installed {name} {pack.version} at {where}")
    msg = (f"{name} is installed. Restart VideoHighlighter to use it."
           if restart else f"{name} is installed.")
    return PackResult(True, msg, name, restart_required=restart)


def discard_download(name: str, lock: Optional[dict] = None,
                     root: Optional[str] = None) -> None:
    """Throw away a paused download (a 'cancel for good' in the UI)."""
    lock = load_lock() if lock is None else lock
    pack = lock.get(name)
    if pack:
        _remove(os.path.join(root or target_root(), STAGING_DIRNAME, pack.asset + ".part"))
