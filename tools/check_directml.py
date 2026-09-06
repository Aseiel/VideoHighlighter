"""Report whether DirectML is usable on this machine, and prove it is not lying.

DirectML's failure modes are quiet ones. It reports a device and runs, so
"it works" is easy to believe while the run is slower than the CPU it replaced.
Three separate things can be true at once and look identical from outside:

  1. It picked the wrong adapter. DirectML enumerates *every* DX12 adapter --
     the integrated GPU, and Microsoft's software renderer -- so device 0 is
     not necessarily the graphics card you bought.
  2. It has the card but not its memory. A backend that sees a fraction of the
     real VRAM swaps to disk on every batch, which shows up as a saturated
     system, an idle GPU, and no error anywhere. This is the single most
     commonly reported DirectML problem and it costs an order of magnitude.
  3. It has the card and the memory and is simply slower than the CPU for the
     op mix in question, because DirectML falls back per operator.

So this answers, on whatever machine it runs on:

  * Is torch-directml installed, does it import, and is the torch beside it the
    version it was built against?
  * Which adapters does it see, and which one would the app take?
  * Does a real forward pass produce finite numbers that agree with the CPU?
  * How much memory can it actually allocate, in one block?
  * Is it faster than this machine's CPU, and by how much?

    python -m tools.check_directml
    python -m tools.check_directml --size 2048     # bigger matmul
    python -m tools.check_directml --skip-benchmark

Exits non-zero when DirectML is unusable, disagrees with the CPU reference, or
turns out to be slower than the CPU -- so it works as a smoke test on an AMD
machine, not just as a printout.

Needs torch and torch-directml. Nothing else in the app is imported beyond
`modules.directml_device`, which is stdlib-only, so this runs in the isolated
DirectML virtualenv that docs/AMD-GPU.md recommends.
"""
from __future__ import annotations

import argparse
import os
import platform
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules import directml_device as dml  # noqa: E402

# Cosine between the DirectML result and an fp32 CPU reference. DirectML is
# free to reassociate and to use fp16 intermediates, so this is looser than the
# CLIP checker's 0.999 -- but a genuinely mangled result lands nowhere near it.
AGREEMENT_MIN = 0.99

# Below this, DirectML is not worth using for the op it was measured on. Set at
# parity rather than at a speedup because the honest answer on a weak card with
# a strong CPU is "use the CPU", and a checker that never says so is decoration.
SPEEDUP_MIN = 1.0


def _hr(title: str) -> None:
    print(f"\n{title}\n" + "-" * len(title))


def report_env() -> bool:
    """Environment and install. False if DirectML cannot be used at all."""
    _hr("Environment")
    print(f"python         {platform.python_version()} ({platform.machine()})")
    print(f"os             {platform.system()} {platform.release()}")

    try:
        import torch
        print(f"torch          {torch.__version__}")
    except Exception as e:
        print(f"torch          NOT IMPORTABLE ({type(e).__name__}: {e})")
        return False

    try:
        import torch_directml
        print(f"torch-directml {getattr(torch_directml, '__version__', '(unknown)')}")
    except Exception as e:
        print(f"torch-directml NOT INSTALLED ({type(e).__name__}: {e})")
        print("\n  pip install torch-directml")
        print("  Note there is no torch-directml>=1.13 - releases are dated dev")
        print("  builds (0.2.5.dev240914), so a normal-looking version floor")
        print("  matches nothing. Install it in its own virtualenv: it pins an")
        print("  exact torch and pip will replace a +xpu or +cu124 build with a")
        print("  stock wheel to satisfy that, silently removing Arc/CUDA support.")
        return False

    print(f"{dml.MODE_ENV:<14} {os.environ.get(dml.MODE_ENV, '(not set)')} "
          f"-> mode={dml.mode()}")
    print(f"{dml.FP16_ENV:<14} {os.environ.get(dml.FP16_ENV, '(not set)')} "
          f"-> fp16={dml.prefer_float16()}")
    return True


def report_adapters() -> bool:
    """Which adapters DirectML sees, and which the app would take."""
    _hr("Adapters")
    probe = dml.probe(refresh=True)
    if not probe.available:
        print(f"unusable: {probe.reason}")
        return False

    for i, name in enumerate(probe.names):
        marker = "  <- the app would use this one" if i == probe.preferred_index else ""
        if dml._is_software_adapter(name):
            marker += "  (software renderer, not a GPU)"
        print(f"  [{i}] {name}{marker}")

    if probe.preferred_index != 0:
        print("\n  Note: adapter 0 is a software renderer and is skipped. That is")
        print("  the intended behaviour — running on it would be slower than the")
        print("  CPU path it replaced, with nothing reporting a fault.")

    print(f"\ntorch device   {probe.device_string()} (backend {probe.backend!r})")
    return True


def report_memory() -> None:
    """The largest single allocation that succeeds, doubling until it does not.

    Reported because DirectML seeing a fraction of the card's memory is the
    quiet fault that costs the most: it does not error, it swaps. A card
    advertised at 8 GB that stops here at 1 GB explains a run that saturates
    the system while the GPU sits idle -- and, on the reports where that has
    been chased down, it was a memory-limiting launch flag (a --lowvram-style
    option) doing it, not the driver.
    """
    _hr("Usable memory")
    import torch

    device = dml.torch_device()
    largest = 0
    mb = 64
    while mb <= 16384:
        try:
            block = torch.empty(mb * 1024 * 1024 // 4, dtype=torch.float32,
                                device=device)
            del block
            largest = mb
        except Exception:
            break
        mb *= 2
    if largest:
        print(f"largest single fp32 allocation that succeeded: ~{largest} MB")
    else:
        print("could not allocate even 64 MB - the device is not really usable")
    print("Compare with the card's advertised VRAM. A figure far below it means")
    print("DirectML is working in a fraction of the memory and will swap.")


def check_forward(size: int) -> bool:
    """A real matmul on DirectML, checked against an fp32 CPU reference."""
    _hr("Correctness")
    import torch

    device = dml.torch_device()
    torch.manual_seed(0)
    a = torch.randn(size, size)
    b = torch.randn(size, size)

    reference = (a @ b).flatten()
    try:
        got = (a.to(device) @ b.to(device)).cpu().flatten()
    except Exception as e:
        print(f"FAILED: the matmul did not run ({type(e).__name__}: {e})")
        print("An 'operator is not currently implemented' message here is")
        print("DirectML's operator coverage, not a bug in the app.")
        return False

    if not torch.isfinite(got).all():
        print("FAILED: the result contains inf/nan")
        return False

    cosine = float(torch.nn.functional.cosine_similarity(
        got.double().unsqueeze(0), reference.double().unsqueeze(0)).item())
    verdict = "ok" if cosine >= AGREEMENT_MIN else "FAILED"
    print(f"cosine vs fp32 CPU reference: {cosine:.6f}  ({verdict}, "
          f"need >= {AGREEMENT_MIN})")
    return cosine >= AGREEMENT_MIN


def benchmark(size: int, iterations: int) -> bool:
    """DirectML against this machine's CPU on the same matmul."""
    _hr("Speed")
    import torch

    device = dml.torch_device()
    a = torch.randn(size, size)
    b = torch.randn(size, size)

    def timed(x, y, sync) -> float:
        # One untimed pass: the first call compiles shaders and allocates, and
        # timing that instead of the steady state overstates the cost several-fold.
        (x @ y)
        sync()
        t0 = time.perf_counter()
        for _ in range(iterations):
            (x @ y)
        sync()
        return (time.perf_counter() - t0) / iterations

    cpu_s = timed(a, b, lambda: None)
    da, db = a.to(device), b.to(device)
    # DirectML queues work asynchronously, so a timer that does not force
    # completion measures how fast Python can submit, not how fast the GPU runs.
    dml_s = timed(da, db, lambda: (da @ db).cpu())

    print(f"matmul {size}x{size}, {iterations} iterations")
    print(f"  cpu        {cpu_s * 1000:8.1f} ms")
    print(f"  directml   {dml_s * 1000:8.1f} ms")
    speedup = cpu_s / dml_s if dml_s else 0.0
    print(f"  speedup    {speedup:8.2f}x  (need >= {SPEEDUP_MIN:.1f}x to be worth it)")
    if speedup < SPEEDUP_MIN:
        print("\n  DirectML is not beating the CPU here. That is a legitimate")
        print(f"  outcome on a weak card - set {dml.MODE_ENV}=off and lose nothing.")
    return speedup >= SPEEDUP_MIN


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--size", type=int, default=1024,
                    help="matmul edge length for the correctness and speed checks")
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--skip-benchmark", action="store_true")
    ap.add_argument("--skip-memory", action="store_true")
    args = ap.parse_args(argv)

    if not report_env():
        return 2
    if not report_adapters():
        return 2
    if not args.skip_memory:
        report_memory()
    ok = check_forward(args.size)
    if ok and not args.skip_benchmark:
        ok = benchmark(args.size, args.iterations)

    _hr("Verdict")
    print("DirectML is usable" if ok else "DirectML is NOT usable as configured")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
