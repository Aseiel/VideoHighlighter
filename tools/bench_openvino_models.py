"""Speed and output of the app's OpenVINO models, so a runtime bump can be compared.

    python -m tools.bench_openvino_models --out before.json --save-outputs before.npz
    (bump the runtime)
    python -m tools.bench_openvino_models --out after.json --compare before.npz

Each model is compiled the way the app compiles it (default config, one
synchronous request) and fed the same seeded input, so both the timings and the
raw outputs are comparable across versions. Models that are not installed are
listed under "missing" and skipped. docs/BENCHMARKS.md says what counts as a
pass when a runtime is bumped.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# (name, path under models/): what the app loads.
MODELS = (
    ("yolox_s", "yolox/yolox_s.xml"),
    ("yolox_tiny", "yolox/yolox_tiny.xml"),
    # Action recognition since 0.13.1 (the Intel encoder/decoder it replaced
    # are measured in docs/BENCHMARKS.md's 2026.2.1 -> 2026.4.0 table).
    ("siglip2_vision", "siglip2-base-patch16-256/vision.onnx"),
)


def _static_shape(partial) -> list:
    return [d.get_length() if d.is_static else 1 for d in partial]


def _inputs(model, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    for port in model.inputs:
        shape = _static_shape(port.get_partial_shape())
        dtype = port.get_element_type().to_dtype()
        out[port.get_any_name()] = rng.random(shape, dtype=np.float32).astype(dtype)
    return out


def bench(core, name: str, xml: Path, device: str, runs: int, warmup: int, seed: int):
    model = core.read_model(str(xml))
    for port in model.inputs:
        if port.get_partial_shape().is_dynamic:
            model.reshape({port.get_any_name(): _static_shape(port.get_partial_shape())})
    t0 = time.perf_counter()
    compiled = core.compile_model(model, device)
    compile_s = time.perf_counter() - t0
    request = compiled.create_infer_request()
    feed = _inputs(model, seed)
    for _ in range(warmup):
        request.infer(feed)
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        request.infer(feed)
        times.append((time.perf_counter() - t) * 1000.0)
    outputs = {f"{name}@{device}:{i}": np.array(request.get_output_tensor(i).data)
               for i in range(len(compiled.outputs))}
    times.sort()
    return {
        "model": name, "device": device,
        "compile_s": round(compile_s, 3),
        "median_ms": round(statistics.median(times), 3),
        "p90_ms": round(times[math.ceil(0.9 * len(times)) - 1], 3),
        "fps": round(1000.0 / statistics.median(times), 1),
    }, outputs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", default=str(ROOT / "models"))
    ap.add_argument("--devices", nargs="+", default=None, help="default: every device found")
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", help="write the results as JSON")
    ap.add_argument("--save-outputs", help="write the raw outputs (.npz) to compare later")
    ap.add_argument("--compare", help="an .npz from --save-outputs: report how far outputs moved")
    args = ap.parse_args(argv)

    import openvino as ov
    core = ov.Core()
    devices = args.devices or [d for d in core.available_devices if d.startswith(("CPU", "GPU"))]
    result = {
        "openvino": ov.__version__,
        "devices": {d: core.get_property(d, "FULL_DEVICE_NAME").strip() for d in devices},
        "runs": args.runs, "rows": [], "missing": [],
    }
    all_outputs = {}
    for name, rel in MODELS:
        xml = Path(args.models) / rel
        if not xml.is_file():
            result["missing"].append(str(xml))
            continue
        for device in devices:
            try:
                row, outs = bench(core, name, xml, device, args.runs, args.warmup, args.seed)
            except Exception as exc:  # noqa: BLE001 - one device failing is a result too
                row, outs = {"model": name, "device": device, "error": f"{type(exc).__name__}: {exc}"}, {}
            result["rows"].append(row)
            all_outputs.update(outs)
            print(json.dumps(row), flush=True)

    if args.compare:
        before = np.load(args.compare)
        diffs = {}
        for key, now in all_outputs.items():
            if key not in before:
                continue
            was = before[key].astype(np.float64)
            now = now.astype(np.float64)
            scale = float(np.abs(was).max()) or 1.0
            diffs[key] = {"max_abs": float(np.abs(now - was).max()),
                          "max_rel_to_range": float(np.abs(now - was).max() / scale)}
        result["output_drift"] = diffs
    if args.save_outputs:
        np.savez_compressed(args.save_outputs, **all_outputs)
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
