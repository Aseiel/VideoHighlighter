"""Time the exported SigLIP2 files (export_siglip.py) the way the app would run them.

    python tools/teach_lab/bench_siglip_ir.py <export dir> [--seconds 6] [--devices GPU,CPU] [--clip]

Image tower, OpenVINO IR on each device (default inference precision: f16 on
the GPU, f32 on the CPU... unless the CPU has native bf16/f16), at 1, 8 and 32
frames per call, latency hint and throughput hint (4 requests in flight for
the latter); ONNX Runtime CPU on the .onnx. Text tower: one prompt per call
(a typed query), OpenVINO. --clip adds the app's CLIP ViT-B/32 image tower
converted the same way, as the reference.

Model time only: random input, after warm-up. A window is 8 frames.
"""
import argparse
import os
import platform
import subprocess
import sys
import time

import numpy as np


def cpu_name():
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "(Get-CimInstance Win32_Processor).Name"], capture_output=True, text=True)
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
    return (time.perf_counter() - t0) / n


def bench_ov(core, model, dev, shape, seconds, dtype=np.float32, hint="LATENCY"):
    cfg = {"PERFORMANCE_HINT": hint}
    net = core.compile_model(model, dev, cfg)
    x = (np.random.rand(*shape) * 2 - 1).astype(dtype) if dtype == np.float32 else \
        np.random.randint(0, 1000, shape).astype(dtype)
    if hint == "LATENCY":
        req = net.create_infer_request()
        return timed(lambda: req.infer([x]), seconds)
    import openvino as ov
    q = ov.AsyncInferQueue(net, 4)

    def run():
        for _ in range(4):
            q.start_async([x])
        q.wait_all()
    return timed(run, seconds) / 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--seconds", type=float, default=6)
    ap.add_argument("--devices", default="GPU,CPU")
    ap.add_argument("--clip", action="store_true")
    args = ap.parse_args()
    import json

    import openvino as ov
    core = ov.Core()
    meta = json.load(open(os.path.join(args.dir, "siglip.json"), encoding="utf-8"))
    s = meta["image_size"]
    models = [("SigLIP2 base image tower", os.path.join(args.dir, "vision.xml"), s)]
    if os.path.exists(os.path.join(args.dir, "vision_int8.xml")):
        models.append(("SigLIP2 base image INT8", os.path.join(args.dir, "vision_int8.xml"), s))
    if args.clip:
        import torch
        from transformers import CLIPVisionModelWithProjection

        class Wrap(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, pixel_values):
                return self.m(pixel_values=pixel_values).image_embeds
        m = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").eval()
        clip_ir = ov.convert_model(Wrap(m), example_input=torch.randn(1, 3, 224, 224))
        clip_xml = os.path.join(args.dir, "_clip_b32_vision.xml")
        ov.save_model(clip_ir, clip_xml, compress_to_fp16=True)
        models.append(("CLIP ViT-B/32 image tower", clip_xml, 224))

    print(f"CPU: {cpu_name()}")
    for d in core.available_devices:
        print(f"{d}: {core.get_property(d, 'FULL_DEVICE_NAME')}")
    print(f"OpenVINO {ov.__version__}\n")
    print(f"{'model':28} {'runtime':34} {'frames/call':>11} {'ms/frame':>9} {'ms/window':>10} {'frames/s':>9}")
    for label, xml, size in models:
        for dev in args.devices.split(","):
            if dev not in core.available_devices:
                continue
            prec = core.get_property(dev, "INFERENCE_PRECISION_HINT")
            prec = prec.get_type_name() if hasattr(prec, "get_type_name") else str(prec)
            for hint in ("LATENCY", "THROUGHPUT"):
                for n in (1, 8, 32):
                    if hint == "THROUGHPUT" and n == 1:
                        continue
                    model = core.read_model(xml)
                    model.reshape([n, 3, size, size])
                    dt = bench_ov(core, model, dev, (n, 3, size, size), args.seconds, hint=hint)
                    print(f"{label:28} {f'OpenVINO {dev} {prec} {hint.lower()}':34} {n:11} "
                          f"{dt / n * 1000:9.2f} {dt / n * 8000:10.1f} {n / dt:9.0f}", flush=True)
        onnx = xml.replace(".xml", ".onnx")       # vision_int8 has no .onnx
        if os.path.exists(onnx) and "CPU" in args.devices:
            import onnxruntime as ort
            sess = ort.InferenceSession(onnx, providers=["CPUExecutionProvider"])
            for n in (1, 8, 32):
                x = (np.random.rand(n, 3, size, size) * 2 - 1).astype(np.float32)
                dt = timed(lambda: sess.run(None, {"pixel_values": x}), args.seconds)
                print(f"{label:28} {'ONNX Runtime CPU fp32':34} {n:11} {dt / n * 1000:9.2f} "
                      f"{dt / n * 8000:10.1f} {n / dt:9.0f}", flush=True)
    for tname in ("text.xml", "text_int8w.xml"):
        txt = os.path.join(args.dir, tname)
        if not os.path.exists(txt):
            continue
        for dev in args.devices.split(","):
            if dev in core.available_devices:
                model = core.read_model(txt)
                model.reshape([1, meta["text_length"]])
                dt = bench_ov(core, model, dev, (1, meta["text_length"]), args.seconds, dtype=np.int64)
                print(f"{'SigLIP2 base ' + tname:28} {f'OpenVINO {dev}':34} {'1 prompt':>11} "
                      f"{dt * 1000:9.2f}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
