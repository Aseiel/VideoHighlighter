"""PyInstaller runtime hook: put installed packs on the import path.

A pack is a folder of Python packages kept outside the frozen bundle —
``packs/<name>/site-packages`` next to the exe — so the heavy, rarely changing
parts (PyTorch and its CUDA libraries) ship and update separately from the
app. This runs before ``main.py``, in the app and in every spawned child, so
``import torch`` anywhere finds the pack.

Packs are *appended* to ``sys.path``: the frozen bundle is searched first, so a
pack can add packages but never shadow one the app bundles itself (a second
numpy would be the classic way to break everything).

Exactly one pack per ``dep`` is loaded. Two torch builds on one path would mix
a CPU torch's Python with a CUDA torch's DLLs, so when an install holds both
(the NVIDIA pack landed beside the CPU one) the higher ``priority`` in
``pack.json`` wins; a pack without a ``pack.json`` is its own dep.
"""
import json
import os
import sys


def _chosen_packs(root):
    best = {}
    try:
        names = sorted(os.listdir(root))
    except OSError:
        return []
    for name in names:
        site = os.path.join(root, name, "site-packages")
        if not os.path.isdir(site):
            continue
        try:
            with open(os.path.join(root, name, "pack.json"), encoding="utf-8") as fh:
                info = json.load(fh)
        except (OSError, ValueError):
            info = {}
        dep = str(info.get("dep") or name)
        try:
            priority = int(info.get("priority", 0))
        except (TypeError, ValueError):
            priority = 0
        if dep not in best or priority > best[dep][0]:
            best[dep] = (priority, name, site)
    return [best[dep] for dep in sorted(best)]


def _install_packs():
    root = os.path.join(os.path.dirname(sys.executable), "packs")
    for _priority, name, site in _chosen_packs(root):
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
