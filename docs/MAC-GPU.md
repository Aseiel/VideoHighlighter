# Apple GPU (experimental)

On a Mac there is no CUDA, and OpenVINO has no GPU plugin for macOS, so before this
every model ran on the processor. The Mac build now carries ONNX Runtime,
whose **Core ML** provider runs a model on the Apple GPU (through Metal) or the
Neural Engine.

## What moves

The same two models that the AMD/DirectML path moves, through the same code:

| Stage | On a Mac |
|---|---|
| Object detection (stock YOLOX) | Core ML |
| Action recognition (SigLIP2 frame encoder) | Core ML, checked against a reference vector before use |
| Everything else (CLIP, Whisper, motion) | processor |

torch can see the GPU through Metal (MPS) as well, and the log says so, but no
model here uses it yet.

Anything Core ML cannot run falls back to the processor: an operator it lacks
runs on the CPU inside the same session, and a model that will not load at all
leaves detection on OpenVINO and the frame encoder on the processor, with a
line in the log.

## Checking it

The log at the start of a run says what was found:

```
✅ Apple GPU via Core ML (ONNX Runtime 1.24.4) — Apple M2
   object detection and action recognition on the GPU / Neural Engine …
✅ Object detector: YOLOX on CoreMLExecutionProvider (yolox_s.onnx)
```

**Settings → Compute** on a Mac lists only Automatic, "Apple GPU (Core ML)" and
"Processor only" — the PC backends do not exist on macOS. "Processor only" is
how to time a run with and without Core ML.

## Switches (environment variables)

| Variable | Values | Effect |
|---|---|---|
| `VH_COREML` | `off` | Keep ONNX Runtime on the processor |
| `VH_COREML_UNITS` | `all` (default), `gpu`, `ane`, `cpu` | Which silicon Core ML may use: all of it, the GPU only (Metal), the Neural Engine only, or none |

Starting the app from Terminal with, say,
`VH_COREML_UNITS=gpu /Applications/VideoHighlighter.app/Contents/MacOS/VideoHighlighter`
compares the GPU against the Neural Engine on the same clip. Timings from real
Macs are what decides the default, so please share them on Discord or in an
issue.
