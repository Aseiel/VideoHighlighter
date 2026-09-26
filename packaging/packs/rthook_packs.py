"""PyInstaller runtime hook: put installed packs on the import path.

A pack is a folder of Python packages kept outside the frozen bundle —
``packs/<name>/site-packages`` — so the heavy, rarely changing parts (PyTorch
and its CUDA libraries) ship and update separately from the app. This runs
before ``main.py``, in the app and in every spawned child, so ``import torch``
anywhere finds the pack.

Packs are *appended* to ``sys.path``: the frozen bundle is searched first, so a
pack can add packages but never shadow one the app bundles itself (a second
numpy would be the classic way to break everything).

Exactly one pack per ``dep`` is loaded. Two torch builds on one path would mix
a CPU torch's Python with a CUDA torch's DLLs, so when an install holds both
(the NVIDIA pack landed beside the CPU one) the higher ``priority`` in
``pack.json`` wins; a pack without a ``pack.json`` is its own dep.

**Two roots.** Beside the exe, and the per-user folder
(``%LOCALAPPDATA%\\VideoHighlighter``). The installer writes beside the exe;
the app's own downloader (modules/packs/pack_manager.py) does too when it can,
and uses the per-user folder when the exe's folder is read-only. On a tie the
exe's folder wins. The per-user rule must match pack_manager.per_user_dir.

**Maintenance, main process only.** A pack installed while the app runs
cannot replace the one whose DLLs that process has loaded, so pack_manager
leaves two kinds of note for the next start, handled here before anything
imports torch:

* ``packs/<name>.pending`` — a new copy of an installed pack: the old folder
  goes to ``packs/.trash`` and the pending one takes its name;
* ``packs/<name>/.remove`` naming another pack — this one is superseded (the
  CPU torch once the NVIDIA one is in); it goes to the trash, but only if the
  pack that replaces it is really there.

Moving to the trash is a rename, so a folder still held open by another
running instance fails to move and is simply left for a later start. A spawned
worker skips all of this: its parent is running on exactly those files.
"""
import json
import os
import shutil
import sys

_PENDING = ".pending"
_REMOVE = ".remove"
_TRASH = ".trash"


def _roots():
    roots = [os.path.dirname(sys.executable)]
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser(r"~\AppData\Local")
    elif sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Application Support")
    else:
        base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    roots.append(os.path.join(base, "VideoHighlighter"))
    out, seen = [], set()
    for root in roots:
        key = os.path.normcase(os.path.abspath(root))
        if key not in seen:
            seen.add(key)
            out.append(os.path.join(root, "packs"))
    return out


def _is_worker():
    # multiprocessing's spawn children (PyInstaller's freeze_support path).
    return any(a.startswith("--multiprocessing") for a in sys.argv[1:])


def _pack_info(folder):
    try:
        with open(os.path.join(folder, "pack.json"), encoding="utf-8-sig") as fh:
            info = json.load(fh)
        return info if isinstance(info, dict) else {}
    except (OSError, ValueError):
        return {}


def _is_pack(folder):
    return os.path.isdir(os.path.join(folder, "site-packages"))


def _to_trash(root, name):
    """Rename ``root/name`` into ``root/.trash``. False if it is in use."""
    trash = os.path.join(root, _TRASH)
    try:
        os.makedirs(trash, exist_ok=True)
        os.rename(os.path.join(root, name),
                  os.path.join(trash, f"{name}-{os.getpid()}"))
        return True
    except OSError:
        return False


def _maintain(roots):
    for root in roots:
        try:
            names = sorted(os.listdir(root))
        except OSError:
            continue
        for name in names:
            if not name.endswith(_PENDING):
                continue
            pending = os.path.join(root, name)
            target = name[: -len(_PENDING)]
            if not (_is_pack(pending) and os.path.isfile(os.path.join(pending, "pack.json"))):
                continue
            if os.path.exists(os.path.join(root, target)) and not _to_trash(root, target):
                continue
            try:
                os.rename(pending, os.path.join(root, target))
                print(f"[packs] {target}: updated")
            except OSError:
                pass

    present = {n for r in roots for n in _safe_listdir(r) if _is_pack(os.path.join(r, n))}
    for root in roots:
        for name in _safe_listdir(root):
            marker = os.path.join(root, name, _REMOVE)
            if not os.path.isfile(marker):
                continue
            try:
                with open(marker, encoding="utf-8") as fh:
                    replacement = fh.read().strip()
            except OSError:
                continue
            if replacement and replacement != name and replacement in present:
                if _to_trash(root, name):
                    print(f"[packs] {name}: removed, superseded by {replacement}")

    for root in roots:
        shutil.rmtree(os.path.join(root, _TRASH), ignore_errors=True)


def _safe_listdir(root):
    try:
        return sorted(os.listdir(root))
    except OSError:
        return []


def _chosen_packs(roots):
    if isinstance(roots, str):
        roots = [roots]
    best = {}
    for root in roots:
        for name in _safe_listdir(root):
            # .pending, .trash, .partial and the like are never loaded.
            if name.startswith(".") or "." in name:
                continue
            folder = os.path.join(root, name)
            if not _is_pack(folder):
                continue
            info = _pack_info(folder)
            dep = str(info.get("dep") or name)
            try:
                priority = int(info.get("priority", 0))
            except (TypeError, ValueError):
                priority = 0
            if dep not in best or priority > best[dep][0]:
                best[dep] = (priority, name, os.path.join(folder, "site-packages"))
    return [best[dep] for dep in sorted(best)]


def _install_packs():
    roots = _roots()
    if not _is_worker():
        try:
            _maintain(roots)
        except Exception as exc:  # never stop the app from starting over this
            print(f"[packs] maintenance skipped: {exc}")
    for _priority, name, site in _chosen_packs(roots):
        sys.path.append(site)
        # torch registers its own DLL folder on import; this only makes the
        # order independent of which package happens to import torch first.
        lib = os.path.join(site, "torch", "lib")
        if os.name == "nt" and os.path.isdir(lib):
            try:
                os.add_dll_directory(lib)
            except OSError:
                pass
        print(f"[packs] {name}")


_install_packs()
