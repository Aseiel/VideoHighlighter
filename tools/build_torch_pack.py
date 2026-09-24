"""Build a PyTorch pack: torch + torchvision as a folder the frozen app loads.

    python tools/build_torch_pack.py --variant cu128 --out build-packs/torch-cu128
    python tools/build_torch_pack.py --variant cpu   --out build-packs/torch-cpu

The pack holds only what is PyTorch's own. Anything the app also bundles
(numpy, Pillow, typing_extensions, ...) is removed, because the frozen bundle
is searched first and a half-duplicate is how imports break in ways nobody
sees until a customer does. ``KEEP`` is therefore an allow-list, and
``tools/pack_core_args.py`` makes the core bundle the rest whole.

Writes ``pack.json`` beside ``site-packages`` with the pack's identity; the
release job turns those into ``packs.json`` (asset, size, sha256).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

TORCH_VERSION = "2.7.1"
TORCHVISION_VERSION = "0.22.1"
INDEX = "https://download.pytorch.org/whl/{variant}"

# Top-level names that belong to the pack. Everything else pip brings along is
# the core's job (see pack_core_args.SHARED).
KEEP = ("torch", "torchvision", "functorch", "torchgen", "sympy", "mpmath",
        "networkx", "isympy")


def _kept(entry: str) -> bool:
    base = entry.split("-", 1)[0].split(".", 1)[0].lower()
    return base in KEEP


def _size(path: str) -> int:
    return sum(os.path.getsize(os.path.join(d, f))
               for d, _, files in os.walk(path) for f in files)


def build(variant: str, out: str) -> dict:
    site = os.path.join(out, "site-packages")
    if os.path.exists(out):
        shutil.rmtree(out)
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--no-cache-dir", "--quiet",
        "--target", site,
        f"torch=={TORCH_VERSION}", f"torchvision=={TORCHVISION_VERSION}",
        "--index-url", INDEX.format(variant=variant),
        "--extra-index-url", "https://pypi.org/simple",
    ])
    raw = _size(site)

    for entry in os.listdir(site):
        if not _kept(entry):
            path = os.path.join(site, entry)
            (shutil.rmtree if os.path.isdir(path) else os.remove)(path)
    # Build-time only, never loaded by a running app — same strip as the
    # release job applies to the bundled torch.
    for sub in ("include", "test", "share"):
        shutil.rmtree(os.path.join(site, "torch", sub), ignore_errors=True)
    for d, _, files in os.walk(site):
        for f in files:
            if f.endswith(".lib"):
                os.remove(os.path.join(d, f))

    info = {
        "name": f"torch-{variant}",
        "dep": "torch",
        "version": f"{TORCH_VERSION}+{variant}",
        "torchvision": TORCHVISION_VERSION,
        "python": f"cp{sys.version_info.major}{sys.version_info.minor}",
        # When two packs provide the same dep, the loader takes the higher.
        # A GPU build beats the CPU one, so an install that briefly holds
        # both (mid-upgrade, or a half-finished setup) still gets the GPU.
        "priority": 0 if variant == "cpu" else 10,
        "bytes_installed": _size(site),
    }
    with open(os.path.join(out, "pack.json"), "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2)
    print(f"pack {info['name']}: pip {raw / 2**30:.2f} GB -> "
          f"kept {info['bytes_installed'] / 2**30:.2f} GB")
    return info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, help="cu128, cpu, ...")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    build(args.variant, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
