# Intel GPUs, via OpenVINO

Intel is the one vendor here whose GPU path is not experimental. It needs no
extra install, no environment variable and no opt-in: OpenVINO ships in every
build, and an Arc or an integrated Xe is picked up automatically.

**The short version.** Leave the compute backend on *Automatic*. If you want to
pin it, choose *Intel (OpenVINO)*. Do not choose *AMD / any DX12 card
(DirectML)* on an Intel machine except to measure it, because it is the slowest
option on this hardware by a wide margin. The numbers below are why.

## Measured on an Arc A750

**2026-09-11.** A Ryzen 5 5600 with an Intel Arc A750 on Windows 11, in the
app's interpreter (`onnxruntime-directml` 1.24.4, `openvino` 2026.2.1), running
a `yolo11n` ONNX export at 640. Thirty inferences after three warm-ups, timing
the inference call alone:

| backend | per frame |
|---|---|
| OpenVINO, Arc GPU | 3.8 ms |
| OpenVINO, processor | 14.9 ms |
| ONNX Runtime, processor | 26.9 ms |
| ONNX Runtime, DirectML | 73.1 ms |

End to end including letterbox and decode, DirectML took 84.7 ms against 36.1 ms
for the same runtime on the processor. The detections were identical either way,
to the pixel: the provider computes the right answer, slowly.

Three things to read out of that table.

**OpenVINO on the GPU is worth about 4× the same runtime on the processor**, and
about 19× DirectML. This is the whole reason the Intel path is not experimental
and the DirectML one is.

**OpenVINO on the *processor* beats ONNX Runtime on the processor**, by nearly
2×. That is not a GPU result at all. It means an Intel machine that somehow
fails to reach its GPU is still better off on OpenVINO than on the alternative.

**DirectML loses to the CPU on Intel.** See below.

## How the app finds an Intel GPU

Two separate probes can answer for Intel, and which one fires depends on how
torch was built. `detect_best_device()` in `modules/device_utils.py` tries them
in this order: CUDA, then Intel, then DirectML, then the processor.

- **torch's own XPU build.** A `torch 2.5+` `+xpu` wheel exposes `torch.xpu`
  natively, with no `ipex` import. Reported as `Intel XPU (OpenVINO)`.
- **OpenVINO directly.** Asks `openvino.Core()` for a device named `GPU`.
  Reported as `Intel GPU (OpenVINO)`.

**The packaged exe always takes the second path.** The release build installs the
CUDA wheel of torch, and `torch.xpu` exists in any build but reports
`is_available()` False on that one, so the XPU probe never fires inside the exe
even on an Arc machine. OpenVINO drives the GPU instead, which is why the
distinction rarely matters in practice.

It matters in one respect. Both probes set `pytorch_device` to `"cpu"`, so even
where torch *can* see the Arc, the models still go through OpenVINO rather than
through torch. That is deliberate. The models this app runs on Intel exist as
OpenVINO IR and OpenVINO is the faster of the two runtimes on that hardware, as
the table above shows.

## What actually uses it

| Feature | On the Intel GPU | Notes |
| --- | --- | --- |
| Object detection | yes, via OpenVINO IR | The heaviest per-frame stage, so the largest part of the win. Exported on first use — see below. |
| Action recognition (Intel encoder/decoder) | yes | `action-recognition-0001`, which ships as OpenVINO IR. |
| Visual search (CLIP prefilter) | yes | Pre-converted to IR at build time by `tools/export_clip_ov.py`. |
| Video encoding (QSV) | yes, indirectly | `modules/encoder_select.py` reads the vendor and prefers `h264_qsv` / `hevc_qsv`. Nothing to do with machine learning. |
| Action recognition (R3D) | **no** | See below. |
| Face, motion | no | OpenVINO on the processor. |

**Detection reaches the GPU through Ultralytics, not through the app's own
device choice.** A full run exports `yolo11<size>.pt` to an OpenVINO folder the
first time it needs one, then loads that folder. Ultralytics compiles it with
OpenVINO's `AUTO` device, which picks the Arc on its own, which is why the log
says the model is loaded and OpenVINO manages the device. The app's
`openvino_device` field is `"GPU"` and drives the *action* encoder and decoder;
it is not what puts the detector on the card.

One consequence: the on-demand analysis in the viewer uses the OpenVINO folder
only if it is already there. It does not export one. So the first full run is
what creates it.

**R3D stays off on Intel, on purpose.** With the action backend on `auto`, R3D
is enabled only where torch or ONNX Runtime can reach a GPU. Neither can here:
`pytorch_device` is `"cpu"` and the ONNX Runtime DirectML flag is not set on an
Intel branch. R3D would therefore run on the processor, and the OpenVINO decoder
on the Arc beats that comfortably, so `auto` picks the decoder and leaves R3D
alone.

You can still force it. Choosing *R3D + CPU (PyTorch, slow)* does exactly what
the label says on an Intel box, and the label is honest about the cost.

## Why DirectML is present but loses here

DirectML is Microsoft's compute backend on top of Direct3D 12, so it runs on
anything with a DX12 driver, Intel included. It exists in this app for AMD,
where there is no vendor runtime to use instead (see [AMD-GPU.md](AMD-GPU.md)).
The automatic ordering puts Intel ahead of it, so an Intel machine never chooses
it on its own. You have to ask.

The reason it loses is where the fast path actually lives. DirectML does not
implement convolution itself. It asks the graphics driver for a **metacommand**,
a vendor-tuned implementation of that operation for that hardware, and falls
back to its own generic compute shaders when the driver offers none. AMD and
NVIDIA maintain mature metacommand sets, because DirectML is the Windows compute
path their customers use. Intel's answer for compute on Arc is OpenVINO, so
there is much less reason for Intel to invest in the DirectML route, and the
73.1 ms above is what the generic fallback costs.

So the Arc measurement is a poor guide to what DirectML does on AMD. It is a
good guide to what it does on Intel: do not use it.

## Caveats on the measurement above

**The adapter is not certain.** Device ids 0 and 1 both bound and performed the
same, Windows lists only a virtual display device and the Arc, and ONNX Runtime
offers no way to ask a session which adapter it took. `modules/ort_directml.py`
binds adapter 0 and does not filter software adapters, unlike
`modules/directml_device.py`, which skips Microsoft's software renderer
deliberately. So the DirectML row may be measuring something other than the Arc.
It does not change the conclusion for Intel users, since OpenVINO wins either
way, but it does mean the row should not be quoted as *DirectML's* speed.

**One model, one size, one machine.** `yolo11n` at 640 on one Ryzen and one Arc.
The ratios are large enough to act on and not precise enough to extrapolate.

## Where the code is

- `modules/device_utils.py` — the probe order, the two Intel branches, and the
  `DeviceInfo` fields each one sets.
- `modules/compute_backend.py` — the backend setting, including the `intel`
  choice that covers both probes.
- `modules/encoder_select.py` — vendor detection for the QSV video encoders.
- `tools/export_clip_ov.py` — the build-time CLIP conversion to OpenVINO IR.
- `docs/AMD-GPU.md` — the DirectML path, and why it is opt-in everywhere.
