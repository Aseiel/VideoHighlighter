# AMD GPUs, via DirectML (experimental)

> On an **Intel** GPU none of this applies: OpenVINO drives it, it is not
> experimental, and DirectML loses badly there. See [INTEL-GPU.md](INTEL-GPU.md).

An AMD card runs everything in this app on the processor. There is now an
opt-in path that changes that on Windows, and it is genuinely experimental:
the mechanism is in the repo and covered by tests, but **none of it has been
measured on an AMD machine by us.** Treat the numbers you get as the only
numbers, and `tools/check_directml.py` as the way to get them.

## Why not ROCm

ROCm is AMD's own stack and it is the faster answer where it applies. It does
not apply here:

- It is Linux-first. Requiring it would mean telling a Windows user to install
  Linux to use a Windows desktop app.
- AMD dropped consumer GPUs from it early. Polaris (RX 470/570/580) has had no
  support since ROCm 4.5, and the Windows HIP SDK reports "no ROCm-capable
  device detected" on exactly the cards most likely to be sitting in a machine
  that would benefit from this.

ZLUDA (a CUDA translation layer) is the other route people try, and it is a
dependency maze that we would then own.

DirectML is Microsoft's compute backend on top of DirectX 12. Anything with a
DX12 driver can run it — every AMD GPU since Polaris, and Intel and NVIDIA
cards too — on stock Windows with no vendor SDK. It is slower than CUDA or
ROCm and it implements a *subset* of torch's operators, which is why nothing
here is on by default and every consumer keeps the fallback it already had.

## Installing it, and the two traps

```bash
pip install torch-directml
```

**Do that in its own virtualenv.** `torch-directml` pins an exact `torch`
version, and pip satisfies that by *replacing* whatever torch is installed. On
a machine carrying a `+xpu` build (Arc) or a `+cu128` build (CUDA), that
silently removes Arc or CUDA support and the app then looks broken for a reason
that has nothing to do with AMD. `CLAUDE.local.md` describes the same class of
accident from the other direction.

**There is no `torch-directml>=1.13`.** Releases are dated dev builds —
`0.2.5.dev240914` — so a requirement line with a normal-looking version floor
resolves to nothing at all and reports a package that does not exist. Pin an
exact version or none.

Python 3.11 is the safest interpreter for this: the published wheels target it,
and 3.12+ has been reported to need workarounds.

`torch-directml` is MIT. It is deliberately **not** in `requirements.txt` and
**not** in any packaged build — partly because of the destructive install
above, and partly because keeping it out means the project is not
redistributing a dependency it has not audited for that.

## Turning it on

Nothing to configure. With `torch-directml` importable, the app picks DirectML
up automatically **only where the alternative was the processor** — it can
never displace CUDA or an Intel path.

**Advanced → Compute → Prefer** is where to change that. It lists the backends
by the hardware they drive — automatic, NVIDIA (CUDA), Intel (OpenVINO), AMD or
any DX12 card (DirectML), processor only — and naming one puts it ahead of the
automatic order. That is how you measure DirectML against whatever this machine
would otherwise have picked, and the run reports which backend it actually got.

A backend this machine does not have falls back to automatic and says so, so a
config copied between machines costs a line in the log rather than a run.

The setting is saved in `config.yaml` under `compute.backend`, and published
into the environment when the app starts. That last part matters: object
detection runs in worker *processes*, which inherit the environment and no
Python state, so a setting that lived only in the GUI would apply to the window
and quietly not to the work.

The environment variables still exist and still win, which is what a developer
testing a build expects:

| Variable | Values | Effect |
| --- | --- | --- |
| `VH_BACKEND` | `auto` / `cuda` / `intel` / `directml` / `cpu` | The same choice as the setting. Set by hand, it outranks the config file. |
| `VH_DIRECTML` | `off` / `auto` (default) / `force` | DirectML's own older switch, still read. Choosing a backend keeps it in step, so the two cannot disagree. |
| `VH_DIRECTML_FP16` | `1` to enable | Run DirectML models in fp16. Off by default; see *Precision* below. |

Forcing it reaches **both** runtimes, torch's first and ONNX Runtime's as the
fallback. That is deliberate: ONNX Runtime is the only DirectML a packaged
build has, so a force that could not reach it would do nothing at all for the
people most likely to try it.

## Check it before trusting it

```bash
python -m tools.check_directml
```

This exists because DirectML's failure modes are quiet. It reports a device and
runs, so "it works" is easy to believe while the run is slower than the CPU it
replaced. Three different things look identical from outside:

1. **It picked the wrong adapter.** DirectML enumerates *every* DX12 adapter,
   including the integrated GPU and Microsoft's software renderer (WARP, a CPU
   implementation of D3D12). Device 0 is not necessarily the card you bought.
   The app therefore **skips software renderers** when choosing an adapter, and
   treats a machine whose *only* adapter is one as having no DirectML at all —
   running on WARP is slower than the CPU path it replaced, with nothing
   reporting a fault. `VH_DIRECTML=force` overrides that. The checker names
   every adapter and marks the one in use.
2. **It has the card but not its memory.** A backend working in a fraction of
   the real VRAM swaps to disk on every batch: the system saturates, the GPU
   sits idle and cool, and nothing anywhere reports an error. The checker
   allocates in doubling blocks until one fails and prints the largest that
   succeeded — compare it against the card's advertised VRAM. Where this has
   been chased down in other DirectML applications, the cause was usually a
   memory-limiting launch flag (a `--lowvram`-style option, or forced fp32)
   rather than the driver; removing such flags restored full VRAM detection and
   a large speedup. If this app ever grows one, that is the first place to look.
3. **It is simply slower than the CPU here.** DirectML falls back per operator,
   so on a weak card with a strong processor the CPU can win. That is a
   legitimate outcome: set `VH_DIRECTML=off` and lose nothing.

The checker exits non-zero on any of those, so it works as a smoke test rather
than only as a printout.

## Testing it without an AMD card

```bash
python -m tools.simulate_directml --all
```

`tools/simulate_directml.py` fakes the hardware: it installs a stand-in
`torch_directml`, hides this machine's real accelerators so the AMD branch is
the one actually taken, and redirects the simulated device onto the CPU so
tensors genuinely move and genuinely run. Then it prints every decision the app
makes — the pipeline backend, all four `resolve_device` results, the encoder
vendor — for each of ten simulated machines: a working RX 570, one where
DirectML sees 1 GB of an 8 GB card, one where the software renderer enumerates
first, one where the first forward pass hits an unimplemented operator, an
0.1.x install where the backend is called `dml`, and so on.

`--run-diagnostic` runs `tools/check_directml.py` itself under the simulation,
end to end.

**What it proves and what it does not.** It exercises the plumbing: which
backend each probe picks, whether a device string survives normalisation and
backend registration, whether the fallbacks fire, what the encoder chain does.
Those are real defect classes and all of them are otherwise invisible until
somebody with an AMD card runs the app — the software-renderer rule above was
found this way, by a simulation showing the app routing models onto WARP.

It proves nothing about DirectML itself. Speed, memory behaviour and above all
operator coverage — the risk that actually matters — are settled only by real
hardware. Under simulation the "GPU" *is* the CPU, so the diagnostic's speed
check always reports a sub-1x speedup and a red verdict; that is the simulation,
not a finding.

## What actually uses it

| Feature | On DirectML | Notes |
| --- | --- | --- |
| Visual search (CLIP prefilter) | yes | Falls back to OpenVINO/CPU if the load fails. |
| Video encoding (AMF) | yes, indirectly | See *The win that needs no model*. |
| Action recognition (R3D) | yes, verified at load | 3D convolution is DirectML's least certain area, so this is proven, not assumed — see below. |
| Action recognition (Intel encoder/decoder) | no | Deliberate: it is small enough that moving it buys nothing, and it exists only as OpenVINO IR. |
| **Object detection** | **no** | Ultralytics has no DirectML backend. See below. |
| Face, motion | no | Unchanged: OpenVINO on the CPU. |

**R3D action recognition proves itself at load.** `pytorch_device` in
`modules/device_utils.py` carries the DirectML string on an AMD box, which is
what routes R3D there, and `auto` enables it — on AMD there is no faster path
being displaced, because OpenVINO's GPU plugin is Intel-only and that branch is
the processor.

R3D is a 3D CNN, though, and 3D convolution is the least certain corner of
DirectML's operator coverage. So this is not taken on trust:
`R3DModelWrapper._warmup()` runs a real forward pass at the actual clip shape
when the model loads, and moves the model to the CPU if the backend cannot
execute it. The `.cpu()` on that pass is load-bearing — DirectML dispatches
asynchronously, so without it a failing operator surfaces later, somewhere
unrelated, and the fallback never sees it.

The worst case is therefore the behaviour the app already had, plus one line
saying why. The failure lands at load, not an hour into a job.

**The Intel action encoder/decoder stays on OpenVINO.** `action-recognition-0001`
is small enough that moving it would buy nothing, and it ships as OpenVINO IR
only — Open Model Zoo publishes no ONNX for it, so there is no artifact
DirectML could run even if it were worth doing.

**Object detection uses DirectML through ONNX Runtime, not torch.** YOLO
runs through Ultralytics, which has no DirectML backend, so
`resolve_yolo_device()` still answers a DirectML request with `cpu` — handing
Ultralytics `privateuseone:0` would trade a slow run for a failed one. Instead
the detector is exported to ONNX and run under ONNX Runtime's DirectML
execution provider, which is a different runtime with no opinion about torch.

Detection is the heaviest per-frame stage, so this is the larger half of the
win. It is also the half that reaches people who did not install anything: see
the next section.

## The packaged build, and why ONNX Runtime is the one that ships

`torch-directml` cannot be bundled. It pins an exact torch — 2.4.1 for the
current release — and pip satisfies that by replacing whatever torch is there,
so a build carrying it could not also carry the CUDA torch the NVIDIA path
needs. One process, one torch. That is why everything above is a source
install, and why an AMD user running the exe had the processor and a log line
telling them to install a package an exe cannot install.

`onnxruntime-directml` has no such coupling: it declares no torch dependency at
all, so it sits in `requirements.txt` beside the CUDA torch and reaches every
Windows build. The wheels are Windows-only, hence the marker on that line, and
it must stay the *only* onnxruntime in there — `onnxruntime` and
`onnxruntime-gpu` install the same package and overwrite each other.

This is the arrangement AnimeJaNai ships for the same reason: one release, ONNX
models, and a backend per vendor (TensorRT for NVIDIA, DirectML for the rest).
The model format is the common denominator, not the framework.

So there are two DirectML paths, with different reach:

| | torch-directml | ONNX Runtime DirectML |
|---|---|---|
| In the packaged build | no, and cannot be | yes |
| Drives | R3D action recognition, visual search | object detection, R3D action recognition |
| Switch | `VH_DIRECTML` | the same `VH_DIRECTML` |

`modules/device_utils.py` reflects that: when torch has a DirectML device it is
used as before, and when it does not — every exe on a DX12 box — the probe
falls through to a branch that reports `DirectML (ONNX Runtime)` and sets two
flags, `onnx_dml_yolo` and `onnx_dml_torch`, for the two model groups that have
an ONNX export in front of them.

`pytorch_device` stays `"cpu"` on that branch and is not a mistake: it is
*torch's* device, and torch genuinely cannot address the card. What moves is the
model, not the framework. Everything else torch drives — the CLIP prefilter,
OWLv2, motion — has no export yet and is still on the processor, which is why
the flags name specific models rather than saying "torch is accelerated".

Both exports are made once on the machine that needs them and then reused: the
detector's next to its weights (`modules/yolo_onnx.py`), R3D's in an
`onnx-cache` folder under the user-data directory (`modules/r3d_onnx.py`, which
keys the filename on the class count and re-exports when imported custom weights
are newer than the cached graph). Each costs tens of seconds, once.

### R3D's second gate

Two things have to be true before the action model moves, not one. ONNX Runtime
has to have DirectML, and the caller has to have granted permission —
`r3d_onnx_dml`, threaded from `_r3d_flags` and the pipeline's backend choice
down to `R3DModelWrapper(allow_onnx_dml=...)`.

The permission exists because the wrapper cannot tell the two "cpu" cases apart
on its own. *"R3D + CPU (PyTorch, slow)"* is a choice a user can make on a DX12
machine and it has to keep meaning the processor there; an automatic fallback
that landed on the CPU because nothing else was available is the opposite case
and wants the GPU. Only the code that knows which of those happened can say.

The session is then tested rather than trusted, twice over. `load()` runs one
real forward pass at the true clip shape, so an operator DirectML cannot place
fails at load instead of an hour into a job — the same reasoning as the torch
warm-up above. And a session that came back on `CPUExecutionProvider`, which is
what ONNX Runtime quietly does when it cannot initialise the provider it was
asked for, is declined outright: it is no faster than the torch model it would
displace, and torch is the better-tested of the two CPU paths.

### The win that needs no model

Detecting the card at all means `modules/encoder_select.py` can finally answer
"amd" and prefer `h264_amf` / `hevc_amf` when re-encoding clips. That is
hardware video encoding, it has nothing to do with machine learning, and it
works whether or not a single model ever runs on DirectML. It is likely the
largest practical speedup on this list.

The vendor comes from the *adapter name*, not from the backend label: DirectML
runs on any DX12 card, so reading the vendor off the label would pick AMF
encoders on an NVIDIA box that had `VH_DIRECTML=force` set for testing, and
lose nvenc for an unrelated reason.

## Measured, once, on the wrong card

**2026-09-11, and the only numbers anyone here has.** They were taken on an
Intel Arc A750, which is not the hardware this path exists for, so the full
four-way table and what it means for Intel users live in
[INTEL-GPU.md](INTEL-GPU.md). Two rows from it matter here:

| backend | per frame |
|---|---|
| ONNX Runtime, processor | 26.9 ms |
| ONNX Runtime, DirectML | 73.1 ms |

The same runtime, the same `yolo11n` export at 640, the same machine. End to end
through `OnnxDetector` — letterbox and decode included — DirectML took 84.7 ms
against 36.1 ms on the processor. The detections were identical either way, to
the pixel: the provider computes the right answer, slowly.

**Why that is weak evidence about AMD.** DirectML does not implement convolution
itself — it asks the graphics driver for a *metacommand*, a vendor-tuned
implementation, and falls back to its own generic compute shaders when the driver
offers none. AMD and NVIDIA maintain mature metacommand sets because DirectML is
the Windows compute path their customers use. Intel's answer for compute is
OpenVINO, so the Arc number is closer to a measurement of the generic fallback
than of DirectML on hardware that supports it properly. It could still lose on
AMD. It would lose for different reasons and by a different margin.

**The adapter is also not certain.** Device ids 0 and 1 both bound and performed
the same, Windows lists only a virtual display device and the Arc, and ONNX
Runtime offers no way to ask a session which adapter it took. Ids above 1 do not
exist, and ORT falls back to the processor without saying so in any readable way
(the error it prints is mojibake), which is why `session_backend()` reads the
provider back off the live session rather than trusting the request.
`modules/ort_directml.py` binds adapter 0 and does *not* filter software
adapters, unlike `modules/directml_device.py` — so on a machine whose adapter 0
is Microsoft's software renderer, that row is measuring a CPU implementation of
D3D12 rather than a graphics card.

### The open question

On an AMD machine, DirectML is not competing with doing nothing. The fallback
there is OpenVINO on the processor, which is the 14.9 ms column above, and it is
a *good* fallback. So a DirectML path that is 3× slower than the same runtime on
the processor would be a regression dressed as an optimisation, and nothing in
the code currently notices.

The answer, when someone gets to it, is the pattern `R3DModelWrapper._warmup()`
already uses for the torch backend: time a handful of frames on both providers
when the detector loads, keep whichever wins, and log the numbers. It costs
under a second, it turns this assumption into a measurement on each user's own
hardware, and it means the worst case is the speed they already had.

Until then: an AMD user who wants to know should run the same comparison rather
than assume the GPU is helping.

## Precision

fp16 is **off** by default on DirectML, which is the opposite of the CUDA path,
where fp16 is unconditional and measured to be free.

DirectML implements half precision unevenly across operators. A model whose
layers mostly support it but which falls back for one pays a conversion on
every call instead of saving bandwidth, and the result is slower than fp32
while looking like an optimisation. The models routed here are small enough
that fp32 fits the cards this exists for. `VH_DIRECTML_FP16=1` opts in — do it
with `tools/check_directml.py` open, not on principle.

R3D follows the same rule from the other direction: `R3DModelWrapper` keeps
fp16 gated on CUDA specifically, so the `auto` backend requests fp32 on
DirectML. The "R3D + CUDA (NVIDIA GPU)" and "R3D + CPU (PyTorch, slow)" choices
in the main window now each name their own device, so the CPU one means the CPU
on an AMD machine too rather than quietly becoming DirectML.

## When it goes wrong

**"The operator aten::… is not currently implemented for the DirectML
backend."** Operator coverage, not a bug in the app. R3D catches this at
load — its warm-up runs a real forward pass and moves the model to the CPU —
and the CLIP loader falls back the same way. Elsewhere it will surface as a
traceback in `debug.log`.

**A crash under load, or a driver reset.** Windows set to the *High
Performance* power plan together with AMD Adrenalin's workload mode set to
*Compute* has been reported as an unstable combination for sustained DirectML
work; reverting both to their defaults (Balanced, and the *Graphics* workload)
resolved it. Worth trying before suspecting the app.

**A device string that torch rejects.** `import torch_directml` is what
registers the backend with torch — the string `privateuseone:0` means nothing
until it has happened, in *that* process. Every call site here imports it
before the first `.to()`, and `modules/directml_device.py:ensure_backend()` is
the helper for any new one. A device string arriving from a worker process, a
CLI flag or a stale config is the case that catches this out.

**The backend is not called what you expect.** torch-directml 0.1.x called it
`dml`; 0.2.x calls it `privateuseone`. Nothing here hardcodes either — the name
is read off the device object the installed package hands back. Use
`modules.directml_device.device_string()` rather than a literal.

## Where the code is

- `modules/directml_device.py` — everything torch-DirectML-specific: the
  opt-in, the probe, device-string normalisation, backend registration. Imports
  nothing but `os` and `typing`, so any module can use it without dragging in
  machinery.
- `modules/ort_directml.py` — the same questions asked of ONNX Runtime: is the
  provider here, which adapter, what session. Shares the `VH_DIRECTML` switch.
- `modules/r3d_onnx.py` — the R3D export and its ONNX Runtime runner: where the
  cached graph lives, when it is stale, and the two refusals above.
- `modules/onnx_detector.py` — the detector that runs an ONNX export, with its
  own letterbox and NMS and no Ultralytics import, so the Pro edition can use
  it with a different export in front.
- `modules/yolo_onnx.py` — the free edition's export half (Ultralytics does the
  converting), kept apart from the runner for that reason.
- `modules/device_utils.py` — the pipeline-wide decision. DirectML sits after
  the Intel probes and before the CPU fallback.
- `llm/clip_prefilter.py` — visual search, with its own `resolve_device` (it
  explains why it does not use `device_utils`).
- `action_recognition.py` — `R3DModelWrapper`, whose warm-up is the guard.
- `modules/encoder_select.py` — the AMF encoder preference.
- `tools/check_directml.py` — the diagnostic, for a real AMD machine.
- `tools/simulate_directml.py` — the fake AMD machine, for every other one.
- `tests/test_directml_device.py`, `tests/test_directml_routing.py`,
  `tests/test_r3d_directml.py`, `tests/test_directml_simulation.py` — the probe,
  the orderings, R3D's fallback, and the simulator's own restore-everything
  contract. None need an AMD card or the package installed.
- `tests/test_ort_directml.py`, `tests/test_onnx_detector.py`,
  `tests/test_onnx_dml_routing.py` — the provider probe, the box arithmetic
  that replaces Ultralytics' own, and the rule that a present DirectML provider
  never takes work away from CUDA or an Intel GPU. None need ONNX Runtime.
