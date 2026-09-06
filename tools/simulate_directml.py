"""Run the DirectML code paths on a machine that has no AMD GPU.

`modules/directml_device.py` was written around one seam so the hardware could
be faked. This is that fake, driven far enough to be worth something: it
installs a stand-in `torch_directml`, *hides this machine's real accelerators*
so the AMD branch is the one actually taken, and redirects the simulated device
onto a real torch device so tensors genuinely compute. Then it runs the app's
own entry points and prints what each of them decided.

What this can and cannot tell you
---------------------------------
It exercises **the plumbing**: which backend each probe picks, whether a device
string survives normalisation and registration, whether the fallbacks fire,
what the encoder chain does, and whether the diagnostic reports what it should.
Every one of those is a real defect class, and every one of them is invisible
until somebody with an AMD card runs the app.

It cannot tell you **anything about DirectML itself** — not speed, not memory
behaviour, and above all not operator coverage, which is the risk that actually
matters and the one only real hardware settles. A green run here means "the app
would do the right thing with a working DirectML"; it does not mean DirectML
works.

    python -m tools.simulate_directml                      # list scenarios + run the default
    python -m tools.simulate_directml --scenario rx570-lowvram
    python -m tools.simulate_directml --scenario op-gap --run-diagnostic
    python -m tools.simulate_directml --all                # every scenario, routing only

Dev-only, under `tools/`, and imports nothing the app ships.
"""
from __future__ import annotations

import argparse
import contextlib
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules import directml_device as dml  # noqa: E402


# ---------------------------------------------------------------------------
# Scenarios — the machines worth pretending to be
# ---------------------------------------------------------------------------

class Scenario:
    """One simulated machine.

    `real_device` is what the simulated DirectML device is redirected to so the
    tensors are genuine. "cpu" is the honest default: it is the only device
    guaranteed to exist, and using this box's Arc would make the simulation
    faster than the thing it stands in for, which invites exactly the wrong
    conclusion.
    """

    def __init__(self, key, summary, adapters=("AMD Radeon RX 570",),
                 backend="privateuseone", installed=True, available=True,
                 vram_mb=8192, op_gap=False, mode=None, real_device="cpu"):
        self.key = key
        self.summary = summary
        self.adapters = list(adapters)
        self.backend = backend
        self.installed = installed
        self.available = available
        self.vram_mb = vram_mb
        self.op_gap = op_gap
        self.mode = mode
        self.real_device = real_device


SCENARIOS = [
    Scenario("rx570", "8 GB Polaris card, everything working — the happy path"),
    Scenario("rx570-lowvram",
             "the quiet fault: DirectML sees 1 GB of an 8 GB card and swaps",
             vram_mb=1024),
    Scenario("software-renderer",
             "adapter 0 is Microsoft's software renderer, not the graphics card",
             adapters=("Microsoft Basic Render Driver", "AMD Radeon RX 570")),
    Scenario("software-only",
             "no graphics driver at all — every adapter is a software renderer",
             adapters=("Microsoft Basic Render Driver",)),
    Scenario("op-gap",
             "the model moves to the device, then the first forward pass hits "
             "an unimplemented operator",
             op_gap=True),
    Scenario("legacy-backend",
             "torch-directml 0.1.x, where the backend is called 'dml'",
             backend="dml"),
    Scenario("missing", "torch-directml is not installed", installed=False),
    Scenario("no-device",
             "the package imports but reports no DX12 device (old driver)",
             available=False),
    Scenario("disabled", "a working card with VH_DIRECTML=off", mode="off"),
    Scenario("nvidia-forced",
             "VH_DIRECTML=force on an NVIDIA box — must not cost it nvenc",
             adapters=("NVIDIA GeForce RTX 4070",), mode="force"),
]

BY_KEY = {s.key: s for s in SCENARIOS}


# ---------------------------------------------------------------------------
# The stand-in package
# ---------------------------------------------------------------------------

def _fake_torch_directml(scenario: Scenario):
    """A module object shaped like `torch_directml`.

    Only the four things `modules/directml_device.py` is allowed to depend on:
    `is_available`, `device_count`, `device_name`, `device`. Keeping it this
    thin is the point — if this stub ever needs to grow, the module under test
    has started depending on something the real package might not provide.
    """
    mod = types.ModuleType("torch_directml")
    mod.__version__ = "0.2.5.dev240914"
    mod.is_available = lambda: scenario.available
    mod.device_count = lambda: len(scenario.adapters)
    mod.device_name = lambda i: scenario.adapters[i]
    mod.device = lambda i=0: f"{scenario.backend}:{i}"
    return mod


# ---------------------------------------------------------------------------
# Making the simulated device real
# ---------------------------------------------------------------------------

class _DeviceShim:
    """Redirect the simulated DirectML device onto a device torch really has.

    Without this the simulation stops at the string: `.to("privateuseone:0")`
    raises, because no backend is registered under that name, and every code
    path past the device decision goes untested. With it, a model genuinely
    moves and genuinely runs — on the CPU, wearing a DirectML device string.

    The two failure modes worth reproducing are attached here rather than to
    the fake package, because both of them happen *after* the device has been
    handed out, which is precisely what makes them hard to see coming:

    * a memory ceiling far below the card's real VRAM (allocations fail while
      nothing reports a fault), and
    * an unimplemented operator (the move succeeds, the forward pass does not).

    Tagging is by attribute on the tensor or module that was moved. That is
    shallow — a tensor derived from a tagged one is not itself tagged — which
    is fine for the boundaries being simulated and would not be fine for
    anything more ambitious. Not a general-purpose emulator.
    """

    OP_GAP_MESSAGE = ("the operator aten::_upsample_bicubic2d_aa is not "
                      "currently implemented for the DirectML backend")

    def __init__(self, scenario: Scenario, device_string: str):
        self.scenario = scenario
        self.device_string = device_string
        self._undo = []

    # -- helpers ---------------------------------------------------------
    def _is_sim_device(self, value) -> bool:
        return isinstance(value, str) and value.lower().startswith(
            self.scenario.backend.lower())

    def _swap(self, args, kwargs):
        """Replace the simulated device with the real one, wherever it appears."""
        hit = False
        args = list(args)
        for i, a in enumerate(args):
            if self._is_sim_device(a):
                args[i], hit = self.scenario.real_device, True
        if self._is_sim_device(kwargs.get("device")):
            kwargs = dict(kwargs, device=self.scenario.real_device)
            hit = True
        return tuple(args), kwargs, hit

    @staticmethod
    def _requested_bytes(args, kwargs) -> int:
        """Bytes a factory call is asking for, well enough to enforce a cap.

        Only handles the shapes this simulation actually produces (a size, or a
        tuple of sizes, plus an optional dtype). A simulator that guessed
        harder would be a worse simulator: it would be wrong quietly.
        """
        import torch

        shape = args[0] if args else kwargs.get("size")
        if isinstance(shape, int):
            count = shape
        elif isinstance(shape, (tuple, list)) and all(isinstance(d, int) for d in shape):
            count = 1
            for d in shape:
                count *= d
        else:
            count = 0
        for extra in args[1:]:
            if isinstance(extra, int):
                count *= extra
        dtype = kwargs.get("dtype") or torch.float32
        return count * getattr(dtype, "itemsize", 4)

    # -- installation ----------------------------------------------------
    def install(self):
        try:
            import torch
        except Exception as e:
            print(f"  (no torch: {e} — routing only, no tensors will run)")
            return
        import torch.nn as nn

        shim = self
        cap_bytes = (self.scenario.vram_mb or 0) * 1024 * 1024

        # Tensor.to / Module.to — where a device string becomes a real move.
        for owner, name in ((torch.Tensor, "to"), (nn.Module, "to")):
            original = getattr(owner, name)

            def moved(self, *args, _orig=original, **kwargs):
                args, kwargs, hit = shim._swap(args, kwargs)
                out = _orig(self, *args, **kwargs)
                if hit:
                    try:
                        out._sim_on_directml = True
                    except Exception:
                        pass
                return out

            setattr(owner, name, moved)
            self._undo.append((owner, name, original))

        # Factories that take device=, so an allocation can be capped.
        for name in ("empty", "zeros", "ones", "randn", "tensor", "full"):
            original = getattr(torch, name, None)
            if original is None:
                continue

            def made(*args, _orig=original, **kwargs):
                args, kwargs, hit = shim._swap(args, kwargs)
                if hit and cap_bytes:
                    want = shim._requested_bytes(args, kwargs)
                    if want > cap_bytes:
                        raise RuntimeError(
                            f"Could not allocate tensor with {want} bytes. "
                            f"There is not enough GPU video memory available!")
                return _orig(*args, **kwargs)

            setattr(torch, name, made)
            self._undo.append((torch, name, original))

        if self.scenario.op_gap:
            self._install_op_gap(torch, nn)

    def _install_op_gap(self, torch, nn):
        """Raise on the first real work done on the simulated device.

        Deliberately not on the move: DirectML's operator gaps surface on the
        forward pass, hours into a run rather than at load, and code that only
        guards the move is code that has not handled this at all.
        """
        shim = self

        original_call = nn.Module._call_impl

        def called(self, *args, _orig=original_call, **kwargs):
            if getattr(self, "_sim_on_directml", False):
                raise RuntimeError(shim.OP_GAP_MESSAGE)
            return _orig(self, *args, **kwargs)

        nn.Module._call_impl = called
        self._undo.append((nn.Module, "_call_impl", original_call))

        original_matmul = torch.Tensor.__matmul__

        def matmul(self, other, _orig=original_matmul):
            if getattr(self, "_sim_on_directml", False) or \
                    getattr(other, "_sim_on_directml", False):
                raise RuntimeError(shim.OP_GAP_MESSAGE)
            return _orig(self, other)

        torch.Tensor.__matmul__ = matmul
        self._undo.append((torch.Tensor, "__matmul__", original_matmul))

    def remove(self):
        for owner, name, original in reversed(self._undo):
            setattr(owner, name, original)
        self._undo.clear()


# ---------------------------------------------------------------------------
# Hiding the real hardware
# ---------------------------------------------------------------------------

def _hide_real_accelerators(stack):
    """Make this machine look like it has no CUDA, no Arc and no OpenVINO GPU.

    Without this the simulation is worthless on any developer machine here:
    `detect_best_device` finds the A750 long before it reaches DirectML, and
    the AMD branch — the entire point — never executes. Restored on exit.
    """
    try:
        import torch
    except Exception:
        return

    if hasattr(torch, "cuda"):
        stack.enter_context(_patched(torch.cuda, "is_available", lambda: False))
    if hasattr(torch, "xpu"):
        stack.enter_context(_patched(torch.xpu, "is_available", lambda: False))

    try:
        import openvino
    except Exception:
        return

    class CpuOnlyCore:
        available_devices = ["CPU"]

        def get_property(self, device, name):
            return "CPU"

    stack.enter_context(_patched(openvino, "Core", CpuOnlyCore))


@contextlib.contextmanager
def _patched(owner, name, value):
    missing = object()
    original = getattr(owner, name, missing)
    setattr(owner, name, value)
    try:
        yield
    finally:
        if original is missing:
            delattr(owner, name)
        else:
            setattr(owner, name, original)


@contextlib.contextmanager
def simulate(scenario: Scenario, hide_real=True, real_tensors=True):
    """Run the block as if this were `scenario`'s machine."""
    with contextlib.ExitStack() as stack:
        if scenario.installed:
            sys.modules["torch_directml"] = _fake_torch_directml(scenario)
            stack.callback(sys.modules.pop, "torch_directml", None)
        else:
            # A stub that raises on import, so "not installed" is simulated even
            # on a machine where the real package happens to be present.
            stack.enter_context(_patched(
                dml, "_import_torch_directml",
                lambda: (_ for _ in ()).throw(
                    ImportError("No module named 'torch_directml'"))))

        old_mode = os.environ.get(dml.MODE_ENV)
        if scenario.mode is None:
            os.environ.pop(dml.MODE_ENV, None)
        else:
            os.environ[dml.MODE_ENV] = scenario.mode
        stack.callback(lambda: (os.environ.pop(dml.MODE_ENV, None) if old_mode is None
                                else os.environ.__setitem__(dml.MODE_ENV, old_mode)))

        if hide_real:
            _hide_real_accelerators(stack)

        dml.set_mode(None)
        probe = dml.probe(refresh=True)
        stack.callback(dml.set_mode, None)

        shim = None
        if real_tensors and probe.available:
            shim = _DeviceShim(scenario, probe.device_string())
            shim.install()
            stack.callback(shim.remove)

        # Caches that would otherwise carry an answer across scenarios.
        try:
            from modules import encoder_select as es
            old_vendor = es._vendor_cache
            es._vendor_cache = es._UNSET
            stack.callback(setattr, es, "_vendor_cache", old_vendor)
        except Exception:
            pass

        yield probe


# ---------------------------------------------------------------------------
# What the app decides, under the simulation
# ---------------------------------------------------------------------------

def _line(label, value):
    print(f"  {label:<34} {value}")


def report_routing(scenario: Scenario) -> None:
    """Every decision the app makes about this machine, in one place."""
    from modules import device_utils as du
    from modules import encoder_select as es
    import llm.clip_prefilter as cp

    probe = dml.probe()
    _line("DirectML probe", repr(probe))
    if not probe.available:
        _line("reason", probe.reason)

    info = du.detect_best_device(log_fn=lambda *a, **k: None)
    _line("pipeline backend", info.backend_name)
    _line("  gpu_available", info.gpu_available)
    _line("  dml_device", info.dml_device)
    _line("  pytorch_device (gates R3D)", info.pytorch_device)
    _line("  openvino_device", info.openvino_device)

    _line("adapters", dml.adapter_names() or "[]")
    # Expected to be "cpu": Ultralytics has no DirectML backend, so the
    # detector's device resolver refuses one on purpose.
    _line("resolve_yolo_device('dml')", du.resolve_yolo_device("dml"))

    _line("CLIP resolve_device('AUTO')", cp.resolve_device("AUTO"))
    _line("CLIP resolve_device('dml')", cp.resolve_device("dml"))
    _line("encoder vendor", es.preferred_gpu_vendor())

    # Action recognition. The viewer and the pipeline share this mapping, so
    # one line covers both — and a disagreement between them would mean a cache
    # built by one is a cache the other would not have produced.
    from modules import analysis_ondemand as ao
    quiet = lambda *a, **k: None  # noqa: E731
    enable, half, device = ao._r3d_flags("auto", log=quiet)
    _line("R3D on 'auto'", f"enabled={enable} fp16={half} device={device}")
    _line("R3D on 'r3d_cpu'", ao._r3d_flags("r3d_cpu", log=quiet))


def exercise_tensors(scenario: Scenario) -> None:
    """Move a real model to the simulated device and run it.

    The part a routing check cannot reach: whether a device string that
    survived every decision above is one torch will actually accept, and
    whether the code around it copes when the device then misbehaves.
    """
    try:
        import torch
        import torch.nn as nn
    except Exception as e:
        print(f"  torch unavailable ({e}) — skipped")
        return

    device = dml.device_string()
    if not device:
        print("  no device to exercise — skipped")
        return

    model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 8))
    try:
        model = model.to(device)
        _line("model.to(device)", f"ok -> {device}")
    except Exception as e:
        _line("model.to(device)", f"FAILED {type(e).__name__}: {e}")
        return

    try:
        out = model(torch.randn(4, 64).to(device))
        _line("forward pass", f"ok, output {tuple(out.shape)}")
    except Exception as e:
        _line("forward pass", f"raised {type(e).__name__}: {e}")
        _line("", "^ this is the class of failure the CLIP loader and R3D's "
                  "warm-up both fall back from")

    # The R3D question, at the shape the wrapper's warm-up actually uses. 3D
    # convolution is the least certain corner of DirectML's operator coverage
    # and the one real reason action recognition might not survive there, so it
    # gets its own line rather than hiding inside the generic forward above.
    conv3d = nn.Conv3d(3, 8, kernel_size=3, padding=1)
    try:
        conv3d = conv3d.to(device)
        clip = torch.zeros(1, 3, 16, 112, 112).to(device)
        with torch.no_grad():
            _line("conv3d at R3D clip shape", f"ok, output {tuple(conv3d(clip).shape)}")
    except Exception as e:
        _line("conv3d at R3D clip shape", f"raised {type(e).__name__}: {e}")
        _line("", "^ R3DModelWrapper._warmup() catches this at load and moves "
                  "action recognition to the CPU")

    # The allocation ceiling, the way tools/check_directml.py measures it.
    largest, mb = 0, 64
    while mb <= 16384:
        try:
            block = torch.empty(mb * 1024 * 1024 // 4, dtype=torch.float32,
                                device=device)
            del block
            largest, mb = mb, mb * 2
        except Exception:
            break
    # The figure is min(simulated cap, what the backing device can really give),
    # so say which bound was hit. Without that, a run where the host's own RAM
    # ran out first reads as a DirectML memory limit that was never simulated.
    bound = ("simulated DirectML cap" if largest and largest * 2 > scenario.vram_mb
             else f"the real {scenario.real_device}, not the simulated cap")
    _line("largest allocation", f"~{largest} MB (cap {scenario.vram_mb} MB; "
                                f"limited by {bound})")


def run_diagnostic() -> int:
    """`tools/check_directml.py`, end to end, under the simulation."""
    from tools import check_directml
    return check_directml.main(["--size", "256", "--iterations", "3"])


# ---------------------------------------------------------------------------

def run_one(scenario: Scenario, args) -> None:
    title = f"{scenario.key} — {scenario.summary}"
    print(f"\n{'=' * len(title)}\n{title}\n{'=' * len(title)}")
    with simulate(scenario, hide_real=not args.keep_real_gpus,
                  real_tensors=not args.no_tensors):
        report_routing(scenario)
        if not args.no_tensors:
            print("\n  -- real tensors on the simulated device --")
            exercise_tensors(scenario)
        if args.run_diagnostic:
            print("\n  -- tools/check_directml.py, under the simulation --")
            code = run_diagnostic()
            print(f"\n  diagnostic exit code: {code}")
            # Said plainly, because a red verdict here is the expected result
            # and would otherwise be read as a finding. The simulated device is
            # the CPU plus this file's dispatch overhead, so it cannot beat the
            # CPU it is being compared against. What the run does establish is
            # that the adapter report, the allocation probe and the correctness
            # check all execute and reach a verdict; the number they reach it
            # with is meaningless until real hardware supplies one.
            print("  (the Speed section is simulation noise — the 'GPU' here IS")
            print("   the CPU, so a sub-1x speedup and a red verdict are correct)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scenario", default="rx570", choices=sorted(BY_KEY),
                    help="which machine to pretend to be")
    ap.add_argument("--all", action="store_true", help="run every scenario")
    ap.add_argument("--run-diagnostic", action="store_true",
                    help="also run tools/check_directml.py under the simulation")
    ap.add_argument("--no-tensors", action="store_true",
                    help="routing decisions only; do not move real tensors")
    ap.add_argument("--keep-real-gpus", action="store_true",
                    help="do not hide this machine's CUDA/XPU/OpenVINO devices "
                         "(the DirectML branch will probably not be reached)")
    args = ap.parse_args(argv)

    # The diagnostic allocates, multiplies and benchmarks. Without the shim
    # redirecting the simulated device onto a real one it cannot do any of that,
    # and it would report "not usable" about the simulation rather than about
    # anything under test — a false negative that looks exactly like a finding.
    if args.run_diagnostic and args.no_tensors:
        print("note: --run-diagnostic needs real tensors; ignoring --no-tensors")
        args.no_tensors = False

    print("Simulated DirectML. This exercises the app's plumbing, not DirectML:")
    print("operator coverage, speed and real VRAM behaviour are settled only by")
    print("an actual AMD card.")

    for scenario in (SCENARIOS if args.all else [BY_KEY[args.scenario]]):
        run_one(scenario, args)

    if not args.all:
        print(f"\nOther scenarios: {', '.join(k for k in sorted(BY_KEY) if k != args.scenario)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
