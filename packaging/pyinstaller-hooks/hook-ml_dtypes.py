"""PyInstaller hook: ship numpy.libs whole, because ml_dtypes needs it whole.

numpy's Windows wheel is repaired with delvewheel, which renames the MSVC
runtime it vendors (msvcp140-<hash>.dll) and puts it in numpy.libs, beside the
package. pandas vendors a byte-identical copy under the same name in
pandas.libs. The frozen build kept that file in pandas.libs only, so numpy.libs
in the bundle was short a DLL that every dev install has.

numpy itself survived that. ml_dtypes did not: onnx imports it, the DirectML
detector's ONNX export (modules/yolo_onnx.py) imports onnx, and the exe died
with "DLL load failed while importing _ml_dtypes_ext". The export failed and
detection stayed on the CPU. Copying the DLL back into numpy.libs by hand was
enough for the export to succeed and detection to run on DmlExecutionProvider,
on a Radeon RX 570.

So this restores the layout a dev install has, rather than chasing which DLL a
given wheel asks for: every file numpy vendors lands in numpy.libs. The names
are globbed, not written down, because the hash changes with numpy releases.
The Windows job checks the result after the build.

Keyed on ml_dtypes rather than numpy on purpose: a hook-numpy.py here would
shadow the hook numpy ships for itself. On macOS there is no numpy.libs and
this collects nothing.
"""
import glob
import importlib.util
import os

_site_packages = os.path.dirname(os.path.dirname(importlib.util.find_spec("numpy").origin))

binaries = [
    (dll, "numpy.libs")
    for dll in glob.glob(os.path.join(_site_packages, "numpy.libs", "*.dll"))
]
