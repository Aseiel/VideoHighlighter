"""PyInstaller hook: ship torchvision's custom-ops library under whatever name it has.

torchvision 0.29 renamed its native ops from _C to _C_stable (and image to
image_stable) and stopped importing them: torchvision/extension.py finds the
file with a FileFinder over the package directory and hands it to
torch.ops.load_library. The contrib hook still names `torchvision._C` as a
hidden import, which no longer exists ("Hidden import torchvision._C not
found!"), so nothing was collected. torchvision then imported fine without its
ops, and the first module that registers against them died instead:
"RuntimeError: operator torchvision::nms does not exist", on `import pipeline`.
The mac build takes whatever torch pip resolves, so it was the first to get 0.29.

So collect every extension-suffixed file in the package root, whatever it is
called, next to torchvision/__init__ where the FileFinder looks. PyInstaller
follows each one's own dylib dependencies (torchvision/.dylibs) from there.

Keyed on torchvision.extension rather than torchvision on purpose: a
hook-torchvision.py here would shadow the contrib hook instead of adding to it.
On an older torchvision this re-collects _C, which the contrib hook already
has at the same destination; the duplicate is dropped.
"""
import importlib.machinery
import importlib.util
import os

_package_dir = os.path.dirname(importlib.util.find_spec("torchvision").origin)

binaries = [
    (os.path.join(_package_dir, name), "torchvision")
    for name in sorted(os.listdir(_package_dir))
    if name.endswith(tuple(importlib.machinery.EXTENSION_SUFFIXES))
]
