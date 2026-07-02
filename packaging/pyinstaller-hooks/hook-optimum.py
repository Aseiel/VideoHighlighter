"""PyInstaller hook: collect optimum (incl. optimum.intel) with .py sources.

optimum/intel/openvino/modeling.py decorates its forward methods with
transformers' add_start_docstrings_to_model_forward, which runs
inspect.getsource() on those (optimum-defined) functions at import time. The
frozen build must be able to serve their module sources or the import dies with
"OSError: could not get source code" before CLIP visual search can load.

There is NO pyinstaller-hooks-contrib hook for optimum (unlike transformers,
whose bundled contrib hook already sets pyz+py and collects dependency
metadata — do NOT shadow it with a hook of our own here). So this hook is the
only place optimum's collection mode gets set.

"pyz+py" ships bytecode + .py sources. The dict lists both the namespace root
and the intel subpackage explicitly; either key alone would cascade to
submodules via PyInstaller's parent-package walk-up, but being explicit avoids
any ambiguity about the optimum namespace package.
"""
module_collection_mode = {
    "optimum": "pyz+py",
    "optimum.intel": "pyz+py",
}
