"""Community model hub for VideoHighlighter.

Share small detection models trained in VideoHighlighter: weights and metadata
only, stored in the author's own Hugging Face account, checked before every use.

Kept byte-identical between the two editions — making and sharing models is
not a paid feature. Nothing here imports Qt at module level except ``gui``,
and ``huggingface_hub`` / ``onnxruntime`` are imported only where used.
"""
from .manifest import HUB_TAG, MANIFEST_NAME, InputSpec, Manifest, ManifestError
from .package import CheckReport, build_package, check_package

__all__ = [
    "HUB_TAG", "MANIFEST_NAME", "InputSpec", "Manifest", "ManifestError",
    "CheckReport", "build_package", "check_package",
]
