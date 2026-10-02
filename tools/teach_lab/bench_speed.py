"""How fast is each encoder per analysis window, on this machine's GPU?

    python tools/teach_lab/bench_speed.py [--windows 4] [--seconds 8] [--only name,name]

Each model gets what it consumes for one window: the Intel encoder and R3D 16
frames, the image encoders 8 frames, V-JEPA 2 16 frames. Inputs are random
tensors of the right shape -- model time only, no decoding or resizing.
Warm-up first, then as many batches as fit in --seconds.

Reports ms per window and frames per second through the model. Torch models
run on XPU in fp16; the Intel encoder and the OpenVINO rows on the OpenVINO GPU
plugin (FP16 inference precision, its default there).
"""
import argparse
import os
import time

import numpy as np
import torch

DEV = "xpu" if torch.xpu.is_available() else "cpu"
DT = torch.float16
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INTEL = os.path.join(ROOT, "models", "intel_action", "encoder", "FP32", "action-recognition-0001-encoder.xml")


def timed(fn, frames_per_window, windows, seconds, sync):
    for _ in range(3):
        fn(); sync()
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        fn(); sync(); n += 1
    dt = (time.perf_counter() - t0) / n
    return dt / windows * 1000, frames_per_window * windows / dt


def torch_sync():
    if DEV == "xpu":
        torch.xpu.synchronize()


def hf_vision(name, size, attr=None):
    from transformers import AutoModel
    m = AutoModel.from_pretrained(name, dtype=DT)
    m = getattr(m, attr) if attr else m
    return m.eval().to(DEV), size


def ov_compile(torch_model, example):
    import openvino as ov
    core = ov.Core()
    om = ov.convert_model(torch_model.float().cpu().eval(), example_input=example.float().cpu())
    return core.compile_model(om, "GPU")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=4, help="windows per batch")
    ap.add_argument("--seconds", type=float, default=8)
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    only = set(args.only.split(",")) - {""}
    W = args.windows
    rows = []

    def want(n):
        return not only or n in only

    # --- Intel action-recognition-0001 encoder, OpenVINO GPU, per frame (as the app runs it)
    if want("intel"):
        import openvino as ov
        core = ov.Core()
        enc = core.compile_model(core.read_model(INTEL), "GPU")
        x1 = np.random.rand(1, 3, 224, 224).astype(np.float32)
        req = enc.create_infer_request()
        ms, fps = timed(lambda: [req.infer([x1]) for _ in range(16 * W)], 16, W, args.seconds, lambda: None)
        rows.append(("Intel encoder (OpenVINO GPU, 1 frame/call)", 16, ms, fps))

    with torch.no_grad():
        # --- 3D CNNs (torchvision), 16 x 112 x 112
        from torchvision.models import video as tvv
        for key, ctor in (("r3d", tvv.r3d_18), ("r21d", tvv.r2plus1d_18)):
            if not want(key):
                continue
            m = ctor(weights=None).eval().to(DEV, DT)
            x = torch.randn(W, 3, 16, 112, 112, device=DEV, dtype=DT)
            ms, fps = timed(lambda: m(x), 16, W, args.seconds, torch_sync)
            rows.append((f"{ctor.__name__} (torch XPU fp16)", 16, ms, fps))
            del m

        # --- image encoders, 8 frames per window
        imgs = [
            ("clip", "openai/clip-vit-base-patch32", 224, "vision_model", "CLIP ViT-B/32"),
            ("siglip_base", "google/siglip2-base-patch16-256", 256, "vision_model", "SigLIP2 base/16 @256"),
            ("siglip", "google/siglip2-so400m-patch14-384", 384, "vision_model", "SigLIP2 so400m/14 @384"),
            ("dino", "facebook/dinov2-large", 224, None, "DINOv2-L/14 @224"),
        ]
        for key, name, size, attr, label in imgs:
            if not want(key):
                continue
            m, _ = hf_vision(name, size, attr)
            x = torch.randn(8 * W, 3, size, size, device=DEV, dtype=DT)
            ms, fps = timed(lambda: m(pixel_values=x), 8, W, args.seconds, torch_sync)
            rows.append((f"{label} (torch XPU fp16)", 8, ms, fps))
            try:
                class Wrap(torch.nn.Module):
                    def __init__(s, inner):
                        super().__init__(); s.inner = inner
                    def forward(s, pixel_values):
                        o = s.inner(pixel_values=pixel_values)
                        return o.pooler_output if getattr(o, "pooler_output", None) is not None else o.last_hidden_state
                cm = ov_compile(Wrap(m), torch.randn(8 * W, 3, size, size))
                xo = np.random.rand(8 * W, 3, size, size).astype(np.float32)
                req = cm.create_infer_request()
                ms, fps = timed(lambda: req.infer([xo]), 8, W, args.seconds, lambda: None)
                rows.append((f"{label} (OpenVINO GPU)", 8, ms, fps))
            except Exception as exc:  # noqa: BLE001
                print(f"  OpenVINO {label}: {type(exc).__name__}: {str(exc)[:120]}")
            del m
            if DEV == "xpu":
                torch.xpu.empty_cache()

        # --- V-JEPA 2 ViT-L, 16 frames @256
        if want("vjepa"):
            from transformers import AutoModel
            m = AutoModel.from_pretrained("facebook/vjepa2-vitl-fpc64-256", dtype=DT).eval().to(DEV)
            x = torch.randn(W, 16, 3, 256, 256, device=DEV, dtype=DT)
            ms, fps = timed(lambda: m.get_vision_features(x), 16, W, args.seconds, torch_sync)
            rows.append(("V-JEPA 2 ViT-L, 16 frames @256 (torch XPU fp16)", 16, ms, fps))

    dev = torch.xpu.get_device_name(0) if DEV == "xpu" else "CPU"
    print(f"\n{dev}, {W} windows per batch, model time only\n")
    print(f"{'model':52} {'frames/window':>13} {'ms/window':>10} {'frames/s':>9}")
    for label, fpw, ms, fps in rows:
        print(f"{label:52} {fpw:13} {ms:10.1f} {fps:9.0f}")


if __name__ == "__main__":
    main()
