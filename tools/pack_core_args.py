"""Extra PyInstaller arguments for a core app that loads PyTorch from a pack.

    $extra = python tools/pack_core_args.py      # one argument per line
    pyinstaller @extra ... main.py

Three rules, each learned the hard way in the first spike:

1. **torch stays out** of the bundle — it arrives as a pack
   (``tools/build_torch_pack.py``, ``packaging/packs/rthook_packs.py``).
2. **The whole standard library goes in.** PyInstaller bundles only what the
   app's own code reaches; with torch excluded nothing reaches ``timeit``, and
   torch's import dies on it.
3. **Packages both sides use go in whole.** The frozen bundle is searched
   before the pack, so a partly-collected Pillow (torchvision needs
   ``ImageDraw``) shadows nothing and fixes nothing — it just fails.
"""
from __future__ import annotations

import importlib.util
import sys

# Lives in the pack (build_torch_pack.KEEP); must not be half-bundled here.
PACK_ONLY = ("torch", "torchvision", "functorch", "torchgen", "sympy",
             "mpmath", "networkx")

# Imported by torch/torchvision AND bundled by the app: bundle them whole.
SHARED = ("numpy", "PIL", "typing_extensions", "filelock", "fsspec",
          "jinja2", "markupsafe", "packaging")

# Packages, collected with their submodules; everything else is one module.
_STDLIB_PACKAGES = {
    "asyncio", "collections", "concurrent", "ctypes", "curses", "dbm", "email",
    "encodings", "html", "http", "importlib", "json", "logging", "multiprocessing",
    "pathlib", "re", "sqlite3", "sysconfig", "tomllib", "unittest", "urllib",
    "wsgiref", "xml", "xmlrpc", "zipfile", "zoneinfo", "compression",
}
_STDLIB_SKIP = {
    "tkinter", "_tkinter", "turtle", "turtledemo", "idlelib", "test",
    "ensurepip", "venv", "lib2to3", "pydoc_data", "this", "antigravity",
    "__phello__", "curses", "_curses", "_curses_panel", "msilib",
}


def _stdlib() -> list:
    out = []
    for name in sorted(sys.stdlib_module_names):
        if name in _STDLIB_SKIP or name.startswith(("_test", "_xx")):
            continue
        try:
            if importlib.util.find_spec(name) is None:   # platform-only module
                continue
        except (ImportError, ValueError):
            continue
        out += (["--collect-submodules", name] if name in _STDLIB_PACKAGES
                else ["--hidden-import", name])
    return out


def args() -> list:
    out = ["--runtime-hook", "packaging/packs/rthook_packs.py"]
    for name in PACK_ONLY:
        out += ["--exclude-module", name]
    for name in SHARED:
        out += ["--collect-submodules", name]
    return out + _stdlib()


if __name__ == "__main__":
    print("\n".join(args()))
