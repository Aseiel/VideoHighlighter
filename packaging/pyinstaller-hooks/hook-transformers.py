"""PyInstaller hook: collect transformers with its .py source files.

transformers' docstring decorators (add_start_docstrings_to_model_forward /
auto_docstring) call inspect.getsource() at import time of every modeling
module. With the default bytecode-only collection that raises
"OSError: could not get source code" inside a frozen build, killing
`from optimum.intel import OVModelForZeroShotImageClassification` (which
imports transformers.models.clip.modeling_clip) before CLIP visual search
can even reach the pre-converted IR. "pyz+py" ships the sources alongside
the bytecode so getsource works, same as the stock torch hook does.

This hook overrides the pyinstaller-hooks-contrib one (additional-hooks-dir
wins), so it re-declares the metadata + data files the stock hook collects.
"""
from PyInstaller.utils.hooks import collect_data_files, copy_metadata

datas = copy_metadata("transformers") + collect_data_files("transformers")
module_collection_mode = "pyz+py"
