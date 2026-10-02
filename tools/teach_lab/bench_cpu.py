"""Every action-model candidate on this machine's processor, one window each.

    python tools/teach_lab/bench_cpu.py [--siglip-dir <dir with siglip2-b32/, siglip2-base/>] [--seconds 5]

Which model is fastest on a processor depends on the processor (AVX2,
AVX-512, VNNI, AMX) and on the precision, so measure it on the machine in
question rather than carry a ranking over from another. Rows:

  Intel action-recognition-0001 encoder, 16 frames: FP32 (what the app loads),
      FP16 and FP16-INT8, one frame per call (as the app calls it) and 16 in one
  r3d_18, 16 frames at 112x112: PyTorch (the app's processor path) and OpenVINO
  SigLIP2 base/32 and base/16 image towers (export_siglip.py, quantize_siglip.py),
      4 and 8 frames, FP16 IR and INT8

Model time only, random input, after warm-up. Missing models are skipped.
"""
import argparse
import os
import platform
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INTEL = os.path.join(ROOT, "models", "intel_action", "encoder")


def cpu_name():
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                             capture_output=True, text=True)
        return out.stdout.strip() or platform.processor()
    except Exception:  # noqa: BLE001
        return platform.processor()


def timed(fn, seconds):
    for _ in range(3):
        fn()
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds or n < 3:
        fn()
        n += 1
    return (time.perf_counter() - t0) / n * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--siglip-dir", default=os.path.join(ROOT, "models"))
    ap.add_argument("--seconds", type=float, default=5)
    args = ap.parse_args()
    import openvino as ov
    core = ov.Core()
    rows = []

    def ov_row(label, xml, shape, per_call=None):
        m = core.read_model(xml)
        if per_call is None:
            m.reshape(list(shape))
        req = core.compile_model(m, "CPU").create_infer_request()
        if per_call:                                   # one frame per call, `per_call` calls
            x = np.random.rand(1, *shape[1:]).astype(np.float32)
            ms = timed(lambda: [req.infer([x]) for _ in range(per_call)], args.seconds)
        else:
            x = (np.random.rand(*shape) * 2 - 1).astype(np.float32)
            ms = timed(lambda: req.infer([x]), args.seconds)
        rows.append((label, ms))
        print(f"  {label:55} {ms:7.1f} ms", flush=True)

    print(f"CPU: {cpu_name()}")
    print(f"OpenVINO {ov.__version__}, CPU capabilities {core.get_property('CPU', 'OPTIMIZATION_CAPABILITIES')}\n")
    for prec in ("FP32", "FP16", "FP16-INT8"):
        xml = os.path.join(INTEL, prec, "action-recognition-0001-encoder.xml")
        if os.path.exists(xml):
            ov_row(f"Intel encoder {prec}, 16 frames, one per call", xml, (16, 3, 224, 224), per_call=16)
            ov_row(f"Intel encoder {prec}, 16 frames in one call", xml, (16, 3, 224, 224))
    try:
        import torch
        from torchvision.models.video import r3d_18
        net = r3d_18().eval()
        xt = torch.randn(1, 3, 16, 112, 112)
        with torch.no_grad():
            ms = timed(lambda: net(xt), args.seconds)
        rows.append(("r3d_18 PyTorch, 16 frames", ms))
        print(f"  {'r3d_18 PyTorch, 16 frames':55} {ms:7.1f} ms", flush=True)
        om = ov.convert_model(net, example_input=xt)
        om.reshape([1, 3, 16, 112, 112])
        req = core.compile_model(om, "CPU").create_infer_request()
        xn = xt.numpy()
        ms = timed(lambda: req.infer([xn]), args.seconds)
        rows.append(("r3d_18 OpenVINO, 16 frames", ms))
        print(f"  {'r3d_18 OpenVINO, 16 frames':55} {ms:7.1f} ms", flush=True)
    except ImportError as exc:
        print(f"  r3d_18 skipped: {exc}")
    for sub, label in (("siglip2-b32", "SigLIP2 base/32"), ("siglip2-base", "SigLIP2 base/16")):
        for f, prec in (("vision.xml", "FP16 IR"), ("vision_int8.xml", "INT8")):
            xml = os.path.join(args.siglip_dir, sub, f)
            if os.path.exists(xml):
                for n in (4, 8):
                    ov_row(f"{label} {prec}, {n} frames", xml, (n, 3, 256, 256))
    return 0


if __name__ == "__main__":
    sys.exit(main())
