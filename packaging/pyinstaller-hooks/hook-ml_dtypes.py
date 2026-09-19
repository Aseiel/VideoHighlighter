"""PyInstaller hook: put the MSVC runtime ml_dtypes links against in numpy.libs.

Wheels repaired with delvewheel rename the MSVC runtime they vendor to
msvcp140-<hash>.dll and keep it in <package>.libs. ml_dtypes' extension links
against that renamed file but ships no copy of its own: it imports numpy first
and relies on numpy.libs, which numpy adds to the DLL search path, to hold it.

It only does when numpy is the version ml_dtypes was built beside. The release
build pins numpy 2.2.6, whose numpy.libs holds msvcp140-2631…; ml_dtypes 0.5.4
wants msvcp140-a4c2…, which numpy 2.4 vendors and which on the build machine
exists only in pandas.libs. The frozen build put it there too, and nowhere on
the search path when ml_dtypes loads. onnx imports ml_dtypes, the DirectML
ONNX exports (modules/yolo_onnx.py, modules/vision/r3d_onnx.py) import onnx, and the
exe died with "DLL load failed while importing _ml_dtypes_ext": the export
failed and everything stayed on the CPU. Copying that DLL into numpy.libs by
hand was enough for the export to succeed and run on DmlExecutionProvider.

An earlier version of this hook copied numpy.libs whole. That cannot help when
the name ml_dtypes asks for is not in numpy.libs to begin with, and 0.11.1 and
0.11.2 shipped broken that way, past a check comparing against the same list.
So the names are read out of the extension itself (delvewheel's names end in a
32-hex-digit hash, so they are unambiguous in the binary) and taken from
whichever *.libs directory has them. A shared name means shared content: the
hash is of the file. The Windows job checks the result after the build.

Keyed on ml_dtypes rather than numpy on purpose: a hook-numpy.py here would
shadow the hook numpy ships for itself. On macOS there are no .libs
directories and this collects nothing.
"""
import glob
import importlib.util
import os
import re

_VENDORED_DLL = re.compile(rb"[\w.+-]+-[0-9a-f]{32}\.dll", re.IGNORECASE)

_site_packages = os.path.dirname(os.path.dirname(importlib.util.find_spec("numpy").origin))


def _vendored_imports(package_dir):
    """Every delvewheel-renamed DLL named inside the package's extensions."""
    names = set()
    for ext in glob.glob(os.path.join(package_dir, "**", "*.pyd"), recursive=True):
        with open(ext, "rb") as f:
            names.update(m.decode("ascii") for m in _VENDORED_DLL.findall(f.read()))
    return names


# numpy.libs first, so a DLL numpy vendors itself is taken from there.
_available = {}
for _dll in glob.glob(os.path.join(_site_packages, "numpy.libs", "*.dll")) + \
        glob.glob(os.path.join(_site_packages, "*.libs", "*.dll")):
    _available.setdefault(os.path.basename(_dll), _dll)

_wanted = {n for n in _available if os.path.dirname(_available[n]).endswith("numpy.libs")}
_wanted |= _vendored_imports(os.path.join(_site_packages, "ml_dtypes"))

for _name in sorted(_wanted - _available.keys()):
    print(f"hook-ml_dtypes: {_name} is imported by ml_dtypes but no *.libs has it")

binaries = [(_available[n], "numpy.libs") for n in sorted(_wanted & _available.keys())]
