# Benchmarks: bumping a runtime

The app's speed and its answers both come from the runtimes under it —
OpenVINO most of all. A new version can be faster, slower, or quietly give
different numbers from the same model, and none of that shows up in a unit
test. So a bump of a major dependency ships only when it is shown to run the
app's models **at the same speed and with the same outputs** as the version it
replaces.

## What counts as a major dependency

Anything that runs a model or decodes the frames fed to one:

| Dependency | Pinned in |
|---|---|
| `openvino` (+ `openvino-tokenizers`, same version) | `requirements.txt`, `build-release.yaml` (twice), `build-packs.yaml` |
| `torch` / `torchvision` | `build-release.yaml` (CPU and CUDA builds), `build-packs.yaml` (the NVIDIA pack) |
| `transformers`, `optimum-intel`, `onnxruntime-directml`, `opencv-python` | not pinned — `requirements.txt` takes the newest (OpenCV: `>=4.7`) |

Every place a pin appears changes in the same PR. Keep the bump in its own
PR, not inside a feature, so a regression points at one change.

The unpinned ones move whenever a release is built. If a release suddenly runs
slower or finds different things, run the benchmark against the previous
release's versions of these first.

## Running it

`tools/bench_openvino_models.py` compiles each model the way the app does
(default config, one synchronous request), feeds it a seeded input, and times
it on every CPU and GPU device OpenVINO finds:

- YOLOX-s and YOLOX-tiny (`models/yolox/`, from `tools/get_yolox_model.py`)
- action recognition: the encoder and decoder the app loads
  (`models/intel_action/.../FP32/`)

```bash
python -m tools.bench_openvino_models --out before.json --save-outputs before.npz
```

Bump the dependency, then:

```bash
python -m tools.bench_openvino_models --out after.json --compare before.npz
```

Run it with nothing else on the GPU — the app, a training run or a browser
playing video will slow it by more than any runtime change. Run each side
**twice** and keep both. Timings of a model that takes a millisecond or two
on a CPU move by a third from one run to the next on their own, so a
difference counts only when every run on one side is faster (or slower) than
every run on the other.

## What passes

- **Outputs:** `output_drift` in `after.json`, `max_rel_to_range` at most
  `1e-3` for every model and device. A patch release usually comes out
  bit-identical (`0.0`). Anything larger needs an explanation in the PR — a
  changed kernel, a new default precision — and a check that detections and
  action scores on a real video did not change.
- **Speed:** the median of every model on every device no more than **10%
  slower** than before. Below 5 ms on a CPU, go by the noise between your own
  two runs instead of the 10%.
- **Compile time:** not more than doubled — it is paid on every start.

Put the before/after table in the PR description with the machine it ran on.

## Reference

OpenVINO 2026.2.1 → 2026.4.0, measured 2026-09-30. AMD Ryzen 5 5600, Intel Arc
A750, 100 runs after 10 of warm-up; median ms of each of two runs per side.

| Model | Device | 2026.2.1 | 2026.4.0 | Reading | Output drift |
|---|---|---|---|---|---:|
| yolox_s | CPU | 47.50 / 47.38 | 52.07 / 49.86 | ~5–10% slower | 0 |
| yolox_s | GPU | 5.16 / 5.30 | 4.95 / 4.99 | ~5% faster | 0 |
| yolox_tiny | CPU | 12.81 / 13.18 | 13.37 / 12.90 | same | 0 |
| yolox_tiny | GPU | 3.50 / 3.44 | 3.45 / 3.10 | same | 0 |
| action encoder FP32 | CPU | 12.01 / 15.44 | 12.28 / 12.11 | same | 0 |
| action encoder FP32 | GPU | 1.50 / 1.34 | 1.51 / 1.50 | same | 0 |
| action decoder FP32 | CPU | 1.39 / 0.98 | 0.68 / 1.44 | same | 0 |
| action decoder FP32 | GPU | 1.24 / 1.09 | 1.08 / 1.22 | same | 0 |

Taking the better run of each side would have shown yolox_tiny on GPU 10%
faster and the decoder on CPU 30% faster — both one lucky run. Read a change
only where every run on one side beats every run on the other.
