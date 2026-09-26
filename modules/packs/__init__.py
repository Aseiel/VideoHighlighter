"""
modules/packs — the optional components (PyTorch for NVIDIA, the CLIP model)
the app downloads on first use instead of carrying them in every download.

Grouping only. ``pack_manager`` is the logic and imports no Qt; ``pack_ui`` is
the thin PySide6 layer over it.
"""
