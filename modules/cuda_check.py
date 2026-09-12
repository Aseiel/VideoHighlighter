"""
cuda_check.py — can this PyTorch actually run on this NVIDIA card?

``torch.cuda.is_available()`` answers whether a driver and a device exist, not
whether this build of torch carries GPU code for that device. A card newer
than the build — an RTX 50-series (compute capability 12.0) under a CUDA 12.4
wheel, whose kernels stop at 9.0 — is "available", and the first real
operation on it fails with "no kernel image is available for execution on the
device". Every consumer then fails or quietly returns nothing, on a machine
that would have run fine on the processor.

``cuda_unusable_reason`` asks the question properly: the build's own
compatibility rule first (no CUDA context needed, so an incompatible card is
never touched), then one tiny operation on the card, the only real proof that
kernels run. Imports nothing at module scope; the answer is cached per process.
"""

from __future__ import annotations

# id(torch module) -> (module, reason). The module is kept so its id cannot be
# reused by another object while the entry exists.
_cache: dict = {}


def _built_archs(cuda) -> list:
    """The ``sm_NN`` compute capabilities this torch build has kernels for."""
    out = []
    for arch in cuda.get_arch_list() or []:
        if isinstance(arch, str) and arch.startswith("sm_"):
            try:
                out.append(int(arch[3:]))
            except ValueError:
                pass
    return out


def _probe(torch) -> str | None:
    cuda = torch.cuda
    if not cuda.is_available():
        return "no CUDA device"

    try:
        built = _built_archs(cuda)
        if built:
            major, minor = cuda.get_device_capability(0)
            # The rule torch itself warns by: compute architectures are
            # backward compatible within a major version, never across one.
            if not any(sm // 10 == major for sm in built):
                return (f"{cuda.get_device_name(0)} (compute capability "
                        f"sm_{major}{minor}) is newer than this PyTorch build "
                        f"supports (sm_{min(built)}–sm_{max(built)}); it needs a "
                        f"PyTorch built for CUDA 12.8 or later")
    except (AttributeError, TypeError, ValueError):
        pass  # a torch that cannot say: let the test operation decide

    try:
        probe = torch.zeros(1, device="cuda")
        probe.add_(1)
        cuda.synchronize()
    except AttributeError:
        return None
    except RuntimeError as e:
        return f"CUDA is present but a test operation on it failed ({e})"
    return None


def cuda_unusable_reason(torch_module=None) -> str | None:
    """None when CUDA code runs on device 0 here, otherwise why it does not.

    Never raises. Pass the torch module in use when there is one (so a caller
    holding a substitute is answered about that); otherwise torch is imported.
    """
    if torch_module is None:
        try:
            import torch as torch_module
        except Exception:  # noqa: BLE001 — no torch means no CUDA
            return "PyTorch is not installed"

    hit = _cache.get(id(torch_module))
    if hit is not None and hit[0] is torch_module:
        return hit[1]
    try:
        reason = _probe(torch_module)
    except Exception as e:  # noqa: BLE001 — a probe must never break the caller
        reason = f"CUDA check failed ({type(e).__name__}: {e})"
    _cache[id(torch_module)] = (torch_module, reason)
    return reason


def cuda_usable(torch_module=None) -> bool:
    """``torch.cuda.is_available()``, answered for the device's real capability."""
    return cuda_unusable_reason(torch_module) is None
