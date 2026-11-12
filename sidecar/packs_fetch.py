"""Fetch the optional packs for the web UI's engine, in the background.

The web build ships the sidecar the way the desktop core ships: without
PyTorch, which arrives as a pack (``packaging/packs/rthook_packs.py``), and
with only the CPU pack inside the archive. The desktop app fetches the NVIDIA
pack and the CLIP model the first time a feature needs them, through a Qt
dialog (``modules/packs/pack_ui.py``). The sidecar has no window to ask in, so
it does the same thing unasked, once, when it starts:

* **CLIP** (``models-clip``) when it is missing: visual search needs it and it
  is the same file for every machine.
* **NVIDIA PyTorch** (``torch-cu128``) only when nvidia-smi finds a card whose
  driver is new enough (``pack_manager.cuda_pack_advice``). Everyone else stays
  on the CPU pack and never downloads 2 GB that would not help.

Everything goes through ``pack_manager``: the lock beside the exe fixes what
may be installed and what it must hash to, an interrupted download resumes on
the next start, and a new PyTorch pack takes effect on the next start (torch
cannot be swapped in a running process). Progress goes to stdout, which the
Tauri shell keeps as the sidecar's log.

Set ``VH_NO_PACK_FETCH=1`` to turn it off (CI smoke tests, metered links).
"""
from __future__ import annotations

import os
import threading
from typing import Callable, List, Optional

DISABLE_ENV = "VH_NO_PACK_FETCH"

LogFn = Callable[[str], None]


def wanted_packs(lock: dict, *, is_ready: Callable[[str, dict], bool],
                 probe: Callable[[], list],
                 advice: Callable[[list], Optional[str]],
                 log: LogFn = print) -> List[str]:
    """The packs this machine should have and does not, in download order."""
    from modules.packs import pack_manager as pm

    out = []
    if pm.CLIP_PACK in lock and not is_ready(pm.CLIP_PACK, lock):
        out.append(pm.CLIP_PACK)
    if pm.CUDA_PACK in lock and not is_ready(pm.CUDA_PACK, lock):
        reason = advice(probe())
        if reason is None:
            out.append(pm.CUDA_PACK)
        else:
            log(f"[packs] {pm.CUDA_PACK} not fetched: {reason}")
    return out


def _progress(name: str, log: LogFn):
    last = {}

    def report(stage: str, done: int, total: int, _label: str) -> None:
        # pack_manager reports every few MB; a line per 10% is enough for a log.
        step = int(10 * done / total) if total else -1
        key = (name, stage)
        if last.get(key) == step:
            return
        last[key] = step
        if total:
            log(f"[packs] {name}: {stage} {100 * done // total}%")
        else:
            log(f"[packs] {name}: {stage}")

    return report


def fetch_missing(*, log: LogFn = print, install=None, lock=None,
                  is_ready=None, probe=None, advice=None) -> list:
    """Install every wanted pack, one after another. Returns the PackResults."""
    from modules.packs import pack_manager as pm

    lock = pm.load_lock() if lock is None else lock
    if not lock:      # running from source, or a build without packs
        return []
    names = wanted_packs(
        lock,
        is_ready=is_ready or pm.is_ready,
        probe=probe or pm.probe_nvidia,
        advice=advice or pm.cuda_pack_advice,
        log=log,
    )
    install = install or pm.install_pack
    results = []
    for name in names:
        log(f"[packs] {name}: downloading ({lock[name].download_mb} MB)")
        result = install(name, lock=lock, progress=_progress(name, log))
        results.append(result)
        if result.ok and result.restart_required:
            log(f"[packs] {name}: installed; it is used from the next start")
        elif result.ok:
            log(f"[packs] {name}: installed")
        else:
            log(f"[packs] {name}: not installed - {result.message}")
    return results


def start_in_background(log: LogFn = print) -> Optional[threading.Thread]:
    """Run ``fetch_missing`` on a daemon thread, unless turned off."""
    if os.environ.get(DISABLE_ENV, "").strip() not in ("", "0"):
        return None

    def run():
        try:
            fetch_missing(log=log)
        except Exception as exc:   # never take the server down over a download
            log(f"[packs] background fetch stopped: {exc}")

    thread = threading.Thread(target=run, name="pack-fetch", daemon=True)
    thread.start()
    return thread
