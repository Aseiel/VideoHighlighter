"""Keep the app's entry script out of its helper processes.

Under the spawn start method, ``multiprocessing`` makes every child import the
parent's ``__main__`` again — from its file path, as ``__mp_main__`` — so that
functions defined there can be unpickled. ``main.py`` defines no process
target (each one lives in an importable module: the thumbnail decoder, the
detection workers, the smoke test), but importing it costs cv2, Qt, OpenVINO,
transformers and the assistant, and prints the start-up lines into the console
once more for every helper the app starts.

A frozen build never paid this: its children stop at ``freeze_support()`` at
the top of ``main.py``. Run from source, each one did, and a viewer action that
starts the thumbnail decoder showed the whole launch again in the console.

``multiprocessing.spawn.get_preparation_data`` passes the main module to the
child by ``__spec__.name`` when it has one, else by ``__file__``. With neither,
the child imports nothing of it.
"""
import sys
from types import ModuleType
from typing import Optional


def keep_main_out_of_children(main_module: Optional[ModuleType] = None) -> None:
    """Stop spawned children from re-running ``main_module`` (default: ``__main__``).

    Call once, in the parent, before the first process starts. Only safe for a
    main module that defines no process target or pickled argument — a child
    could no longer find one there.
    """
    module = main_module if main_module is not None else sys.modules["__main__"]
    if getattr(module, "__spec__", None) is not None:
        module.__spec__ = None
    if hasattr(module, "__file__"):
        del module.__file__
