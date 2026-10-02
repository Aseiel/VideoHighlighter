"""Time exported SigLIP2 image towers on ONNX Runtime DirectML, the route R3D
already takes on a GPU with no CUDA and no Intel path (an AMD card, an old
NVIDIA card).

    python tools/teach_lab/bench_dml.py <vision.onnx> [<vision.onnx> ...] [--frames 8] [--seconds 5]

Needs onnxruntime-directml (the Pro build ships it on Windows). Checks the
DirectML output against ONNX Runtime's CPU on the same input, then times one
window of --frames frames. Model time only, random input, after warm-up.
"""
import argparse
import sys
import time

import numpy as np


def timed(fn, seconds):
    for _ in range(3):
        fn()
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        fn()
        n += 1
    return (time.perf_counter() - t0) / n * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx", nargs="+")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seconds", type=float, default=5)
    args = ap.parse_args()
    import onnxruntime as ort
    if "DmlExecutionProvider" not in ort.get_available_providers():
        print("no DirectML provider: pip install onnxruntime-directml (in its own environment)")
        return 1
    x = np.random.default_rng(0).random((args.frames, 3, args.size, args.size), np.float32) * 2 - 1
    for path in args.onnx:
        ref = ort.InferenceSession(path, providers=["CPUExecutionProvider"]).run(None, {"pixel_values": x})[0]
        so = ort.SessionOptions()
        so.enable_mem_pattern = False                    # DirectML requires both
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        dml = ort.InferenceSession(path, so, providers=["DmlExecutionProvider", "CPUExecutionProvider"])
        out = dml.run(None, {"pixel_values": x})[0]
        cos = ((out * ref).sum(1) / np.linalg.norm(out, axis=1) / np.linalg.norm(ref, axis=1)).min()
        ms = timed(lambda: dml.run(None, {"pixel_values": x}), args.seconds)
        print(f"{path}: DirectML {ms:.1f} ms per {args.frames}-frame window, "
              f"worst cosine vs CPU {cos:.5f}, finite {bool(np.isfinite(out).all())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
