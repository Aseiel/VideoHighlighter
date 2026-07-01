"""PyInstaller hook: collect optimum (incl. optimum.intel) with .py sources.

optimum/intel/openvino/modeling.py decorates its forward methods with
transformers' add_start_docstrings_to_model_forward, which runs
inspect.getsource() on them at import time. The frozen build must be able
to serve these module sources or the import dies with
"OSError: could not get source code" before CLIP visual search can load.
"""
module_collection_mode = "pyz+py"
