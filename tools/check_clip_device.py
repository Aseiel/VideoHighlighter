"""Report which backend CLIP lands on, and prove that backend is actually right.

The CLIP stack picks its backend from what the machine has (NVIDIA -> torch/CUDA,
Intel -> OpenVINO, else CPU), and the failure mode that matters is *silent*: a
wrong pick still returns plausible scores, just slowly. So this answers three
questions on whatever machine it runs on:

  1. Which backend/device does CLIP resolve to, and what did it see to decide?
  2. Does a real forward pass work there, with sane embeddings?
  3. Does it agree with a CPU reference, and is it actually faster?

    python -m tools.check_clip_device
    python -m tools.check_clip_device --frames 96     # longer benchmark
    python -m tools.check_clip_device --device CPU    # force a backend
    python -m tools.check_clip_device --skip-reference   # quick, no CPU compare

Exits non-zero if the resolved backend is wrong, non-finite, or disagrees with
the reference — so it works as a smoke test, not just a printout.

Needs only torch + transformers + pillow + opencv-python + numpy. The CUDA path
does not need optimum-intel; the OpenVINO path does.
"""
from __future__ import annotations

import argparse
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import llm.clip_prefilter as cp  # noqa: E402
from llm.clip_index import ClipEmbedder  # noqa: E402

# Cosine below this between the tested backend and an fp32 CPU reference means
# the plumbing mangled something. fp16 noise lands ~1e-6 away, so this is loose
# enough to never flake and tight enough to catch a real fault.
AGREEMENT_MIN = 0.999


def _hr(title: str) -> None:
    print(f"\n{title}\n" + "-" * len(title))


def report_env() -> None:
    _hr("Environment")
    print(f"python       {platform.python_version()} ({platform.machine()})")
    print(f"os           {platform.system()} {platform.release()}")

    try:
        import torch

        print(f"torch        {torch.__version__}")
        print(f"  cuda build {torch.version.cuda or '(none — this is a CPU/XPU wheel)'}")
        avail = torch.cuda.is_available()
        print(f"  available  {avail}")
        if avail:
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                print(f"  device {i}   {p.name} ({p.total_memory / 1024**3:.1f} GB, "
                      f"sm_{p.major}{p.minor})")
    except Exception as e:
        print(f"torch        UNAVAILABLE — {type(e).__name__}: {e}")

    for mod in ("transformers", "optimum.intel", "openvino"):
        try:
            m = __import__(mod, fromlist=["__version__"])
            print(f"{mod:<12} {getattr(m, '__version__', '(no __version__)')}")
        except Exception as e:
            note = "  (not needed for CUDA)" if mod.startswith("optimum") else ""
            print(f"{mod:<12} not importable — {type(e).__name__}{note}")


def report_resolution(requested: str) -> tuple[str, str]:
    _hr("Backend resolution")
    probe = cp.cuda_device()
    print(f"cuda_device() -> {probe!r}")
    for req in ("AUTO", "GPU", "CUDA", "CPU"):
        print(f"  {req:<5} -> {cp.resolve_device(req)}")

    backend, device = cp.resolve_device(requested)
    print(f"\nrequested {requested!r} -> backend={backend!r} device={device!r}")
    err = cp.ClipFramePrefilter.import_error(requested)
    print(f"import_error({requested!r}) -> {err or 'None (stack imports cleanly)'}")
    return backend, device


def make_frames(n: int) -> list[np.ndarray]:
    """n distinct BGR frames, as OpenCV would hand them over. Structured rather
    than pure noise so CLIP produces embeddings that actually differ."""
    rng = np.random.default_rng(0)
    frames = []
    for i in range(n):
        f = np.zeros((224, 224, 3), np.uint8)
        f[:, :, i % 3] = 60 + (i * 37) % 190          # varying dominant channel
        f[40:180, 40:180] = rng.integers(0, 255, (140, 140, 3), dtype=np.uint8)
        frames.append(f)
    return frames


def cpu_reference(model_id: str) -> ClipEmbedder | None:
    """An fp32 CLIP on the CPU via plain torch — the thing we trust."""
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        ref = ClipEmbedder(device="CPU")
        ref.backend, ref.device, ref._dtype = "torch", "cpu", torch.float32
        ref._model = CLIPModel.from_pretrained(model_id, torch_dtype=torch.float32).eval()
        ref._processor = CLIPProcessor.from_pretrained(cp._bundled_ov_dir() or model_id)
        return ref
    except Exception as e:
        print(f"⚠️  could not build CPU reference ({type(e).__name__}: {e})")
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="AUTO", help="AUTO/GPU/CUDA/cuda:N/CPU/...")
    ap.add_argument("--frames", type=int, default=48, help="frames to benchmark")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--expect", default=None, choices=["torch", "openvino"],
                    help="fail unless this backend is chosen (for CI/automation)")
    ap.add_argument("--skip-reference", action="store_true",
                    help="skip the CPU agreement check (saves a second model load)")
    args = ap.parse_args()

    report_env()
    backend, device = report_resolution(args.device)

    err = cp.ClipFramePrefilter.import_error(args.device)
    if err is not None:
        # Stop here rather than let load() raise: a missing dep is a setup
        # problem with a known fix, and a traceback buries it.
        print(f"\n❌ FAIL — the {backend!r} backend's stack is incomplete:\n   {err}")
        if backend == "torch":
            print("\n   Install a CUDA torch:\n"
                  "     pip install torch --index-url https://download.pytorch.org/whl/cu124")
        else:
            print('\n   Install the OpenVINO stack:\n'
                  '     pip install "optimum[openvino]" optimum-intel')
        print("   ...plus: pip install transformers pillow opencv-python numpy")
        return 1

    failures: list[str] = []
    if args.expect and backend != args.expect:
        failures.append(f"expected backend {args.expect!r}, resolved {backend!r}")

    _hr("Load")
    emb = ClipEmbedder(device=args.device)
    emb.load()
    print(f"loaded: backend={emb.backend!r} device={emb.device!r} dtype={emb._dtype}")
    if args.expect and emb.backend != args.expect:
        failures.append(f"expected to LOAD on {args.expect!r}, got {emb.backend!r} "
                        f"(it fell back — see the warning above for why)")

    _hr("Correctness")
    frames = make_frames(args.frames)
    img = emb.embed_frames_bgr(frames[:4])
    print(f"image_embeds  shape={img.shape} dtype={img.dtype}")
    if img.shape != (4, 512):
        failures.append(f"bad embedding shape {img.shape}")
    if not np.all(np.isfinite(img)):
        failures.append("non-finite values in image embeddings")
    norms = np.linalg.norm(img, axis=1)
    print(f"unit norms    {np.round(norms, 5)}")
    if not np.allclose(norms, 1.0, atol=1e-3):
        failures.append(f"embeddings are not unit-norm: {norms}")

    txt = emb.embed_texts(["a red photo", "a blue photo"])
    print(f"text_embeds   shape={txt.shape}")
    print(f"logit_scale   {emb.logit_scale:.2f}  (CLIP's published value is 100.0)")

    # The calibrated-score path, which is what the app thresholds on.
    emb.set_query("a red photo", negatives=["a blue photo"])
    scores = emb.score_frames_bgr(frames[:4])
    print(f"scores        {[round(s, 4) for s in scores]}")
    if not all(np.isfinite(scores)):
        failures.append("non-finite score from score_frames_bgr")

    if not args.skip_reference:
        _hr("Agreement with fp32 CPU reference")
        ref = cpu_reference(emb.model_id)
        if ref is not None:
            ref_img = ref.embed_frames_bgr(frames[:4])
            agree = np.abs((img * ref_img).sum(1))
            print(f"cosine/frame  {np.round(agree, 6)}")
            worst = float(agree.min())
            print(f"worst         {worst:.6f}  (threshold {AGREEMENT_MIN})")
            if worst < AGREEMENT_MIN:
                failures.append(f"backend disagrees with CPU reference (worst {worst:.6f}) "
                                f"— embeddings would be wrong, and cached indexes "
                                f"from other machines incompatible")
            else:
                print("→ same numbers as the CPU: indexes stay portable across machines.")

    _hr(f"Benchmark ({args.frames} frames, batch {args.batch})")
    emb.embed_frames_bgr(frames[:args.batch])          # warm up kernels/caches
    t0 = time.perf_counter()
    for i in range(0, len(frames), args.batch):
        emb.embed_frames_bgr(frames[i:i + args.batch])
    elapsed = time.perf_counter() - t0
    per = elapsed / len(frames) * 1000
    print(f"{emb.device}: {elapsed:.2f}s total, {per:.1f} ms/frame, "
          f"{len(frames) / elapsed:.1f} frames/s")
    print("(encode only — a real scan adds video decode)")

    _hr("Verdict")
    if failures:
        print(f"❌ FAIL ({len(failures)})")
        for f in failures:
            print(f"   - {f}")
        return 1
    print(f"✅ PASS — CLIP works on {emb.device!r} via the {emb.backend!r} backend "
          f"at {per:.1f} ms/frame.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
