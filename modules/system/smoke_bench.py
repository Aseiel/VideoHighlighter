"""`--smoke-bench <video>`: every model on every route this machine has, timed,
checked against the processor, and repeated to look for leaks.

Nobody on the project has a Mac. The macOS runner does reach Metal (an M1
VM), so CI runs this against the built .app; a tester with real hardware runs
the same command and sends the report back:

    VideoHighlighter --smoke-bench clip.mp4 --smoke-report bench.json

For each model, on each route (Core ML with the GPU and Neural Engine, Core ML
with the GPU only, torch on Metal, the processor):

* how long it takes to load and how many milliseconds an item costs, warm;
* whether its answers match the processor's (cosine of the vectors, or the
  share of words Whisper agrees on), because a fast route that answers
  differently is a broken one;
* how much the process grew to hold it.

Then the same work repeats on the route the app picks, and the process's
memory after each round is the leak check: after the first round (caches,
compiled kernels) it should stop growing. Object detection repeats as the app
runs it, worker processes and all, with 1, 2 and the default number of
workers, so the workers' cost in memory and their gain in speed are measured
rather than assumed.

A report, never a gate: it exits 0 unless the video is missing. Every section
catches its own failure and records it, so one broken route still leaves the
rest of the measurements.
"""
from __future__ import annotations

import gc
import json
import os
import platform
import sys
import time

FRAMES = 32               # frames per timing round
ROUNDS = 10               # rounds of the same work, for the leak check
LEAK_MB_PER_ROUND = 8.0   # growth past the first round that counts as a leak
WHISPER_MODELS = ("base", "small")


def _arg(argv, name, default=None, cast=str):
    if name in argv:
        i = argv.index(name)
        if i + 1 < len(argv):
            return cast(argv[i + 1])
    return default


def _rss_mb() -> float:
    import psutil
    return psutil.Process().memory_info().rss / 2**20


def _settle() -> float:
    gc.collect()
    torch = sys.modules.get("torch")
    try:
        if torch is not None and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:  # noqa: BLE001
        pass
    return _rss_mb()


def _cosine(a, b) -> float:
    import numpy as np
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else 0.0


def _row_cosines(a, b) -> float:
    """The worst cosine between matching rows of two [N, D] arrays."""
    return min(_cosine(x, y) for x, y in zip(a, b))


def _frames(video: str, n: int = FRAMES):
    import cv2
    cap = cv2.VideoCapture(video)
    out = []
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or n
        step = max(1, total // n)
        for i in range(n):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
            ok, frame = cap.read()
            if not ok:
                break
            out.append(frame)
    finally:
        cap.release()
    return out


def _timed(fn, items: int) -> dict:
    """Warm once, then time a second call: ms per item."""
    fn()
    t = time.perf_counter()
    fn()
    return {"ms_per_item": round((time.perf_counter() - t) * 1000 / max(items, 1), 2)}


def _leak(work, rounds: int = ROUNDS) -> dict:
    """RSS after each round of ``work``, and how much it grew past the first."""
    rss = []
    for _ in range(rounds):
        work()
        rss.append(round(_settle(), 1))
    growth = (rss[-1] - rss[0]) / max(len(rss) - 1, 1)
    out = {"rss_mb": rss, "growth_mb_per_round": round(growth, 2),
           "leak": growth > LEAK_MB_PER_ROUND}
    torch = sys.modules.get("torch")
    try:
        if torch is not None and torch.backends.mps.is_available():
            out["mps_allocated_mb"] = round(torch.mps.current_allocated_memory() / 2**20, 1)
            out["mps_driver_mb"] = round(torch.mps.driver_allocated_memory() / 2**20, 1)
    except Exception:  # noqa: BLE001
        pass
    return out


def _section(report, name, fn):
    print(f"⏱ bench: {name}")
    t = time.perf_counter()
    try:
        report[name] = fn()
    except BaseException as e:  # noqa: BLE001 - one broken section must not end the bench
        report[name] = {"error": f"{type(e).__name__}: {e}"}
    report[name]["seconds"] = round(time.perf_counter() - t, 1)
    # Whole, on one line: the log is the copy that survives a failed upload.
    print(f"   {name}: {json.dumps(report[name])}")


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def _environment() -> dict:
    out = {"python": sys.version.split()[0], "machine": platform.machine(),
           "macos": platform.mac_ver()[0] or None, "rss_mb": round(_rss_mb(), 1)}
    try:
        from modules.system.device_utils import _apple_chip_name
        out["chip"] = _apple_chip_name()
    except Exception:  # noqa: BLE001
        pass
    for mod in ("torch", "onnxruntime", "openvino"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception as e:  # noqa: BLE001
            out[mod] = f"missing ({type(e).__name__})"
    try:
        import torch
        out["mps"] = bool(torch.backends.mps.is_available())
    except Exception:  # noqa: BLE001
        out["mps"] = False
    try:
        import onnxruntime as ort
        out["ort_providers"] = ort.get_available_providers()
    except Exception:  # noqa: BLE001
        pass
    return out


def _with_units(units: str):
    """Set Core ML's compute units (and, after a "/", its model format) for
    sessions opened from now on; "" restores the defaults."""
    from modules.system import ort_coreml
    units, _, fmt = (units or "").partition("/")
    for env, value in ((ort_coreml.UNITS_ENV, units), (ort_coreml.FORMAT_ENV, fmt)):
        if value:
            os.environ[env] = value
        else:
            os.environ.pop(env, None)


_COREML_UNITS = (("coreml-all", "all"), ("coreml-gpu", "gpu"),
                 ("coreml-nn-all", "all/neuralnetwork"))


def _frame_encoder(frames) -> dict:
    from modules.vision import frame_encoder as fe
    folder = fe.find_model_dir()
    if folder is None:
        return {"error": "frame encoder not installed"}
    meta = fe.read_meta(folder)
    path = os.path.join(folder, fe.MODEL_FILE)
    pixels = fe.preprocess(frames)
    out, ref = {}, None

    # The NeuralNetwork format ran it at 9.5 s a frame on the M1 runner, so
    # only the ML Program ones; the second "all" load shows the compile cache.
    routes = [("openvino-cpu", fe.OPENVINO_CPU, ""), ("coreml-all", fe.ONNX_GPU, "all"),
              ("coreml-all-again", fe.ONNX_GPU, "all"), ("coreml-gpu", fe.ONNX_GPU, "gpu")]
    for name, route, units in routes:
        before = _settle()
        try:
            _with_units(units)
            t = time.perf_counter()
            runner = fe._open_route(route, path, None)
            fe._check_route(runner, meta["probe"])
            load_s = time.perf_counter() - t
            enc = fe.FrameEncoder(runner, route, folder)
            vecs = enc.encode_pixels(pixels)
            row = {"load_s": round(load_s, 2), "batch": enc.batch,
                   "held_mb": round(_settle() - before, 1),
                   **_timed(lambda: enc.encode_pixels(pixels), len(pixels))}
            if ref is None:
                ref = vecs
            else:
                row["cosine_vs_cpu"] = round(_row_cosines(vecs, ref), 5)
            if route == fe.ONNX_GPU:
                from modules.system import ort_directml
                row["provider"] = ort_directml.session_backend(runner._session)
            out[name] = row
            del enc, runner
        except Exception as e:  # noqa: BLE001
            out[name] = {"error": f"{type(e).__name__}: {e}"}
        finally:
            _with_units("")

    enc = fe.load(log=print)
    if enc is not None:
        out["app_picks"] = enc.label
        # On the processor a round is seconds per frame; fewer frames say as
        # much about growth.
        some = frames if enc.on_gpu else frames[:8]
        out["leak"] = _leak(lambda: enc.encode_bgr(some))
    return out


def _yolox(frames) -> dict:
    import object_recognition
    from modules.system import ort_directml
    from modules.vision import yolox_models
    from modules.vision.detection_backend import YoloxOnnxRuntimeDetector, load_class_names
    path = yolox_models.find_onnx("small")
    names = load_class_names("yolo_objects_labels.json")
    if not path or not names:
        return {"error": "no YOLOX export or labels"}
    out = {"model": os.path.basename(path)}
    variants = [("onnx-cpu", [ort_directml.CPU_PROVIDER], "")]
    variants += [(name, None, units) for name, units in _COREML_UNITS]
    for name, providers, units in variants:
        before = _settle()
        try:
            _with_units(units)
            t = time.perf_counter()
            sess = ort_directml.session(path, providers_override=providers)
            det = YoloxOnnxRuntimeDetector(sess, names)
            det.detect(frames[0])
            row = {"load_s": round(time.perf_counter() - t, 2),
                   "provider": ort_directml.session_backend(sess),
                   "held_mb": round(_settle() - before, 1),
                   **_timed(lambda: [det.detect(f) for f in frames], len(frames))}
            out[name] = row
            del det, sess
        except Exception as e:  # noqa: BLE001
            out[name] = {"error": f"{type(e).__name__}: {e}"}
        finally:
            _with_units("")
    det = object_recognition.directml_detector("small", log=print)
    if det is not None:
        out["leak"] = _leak(lambda: [det.detect(f) for f in frames])
    return out


def _clip(frames) -> dict:
    import numpy as np
    from llm import clip_prefilter as cp
    out, ref = {}, None
    devices = ["cpu"]
    if cp.mps_device():
        devices.append("mps")
    for dev in devices:
        before = _settle()
        try:
            t = time.perf_counter()
            clip = cp.ClipFramePrefilter()
            if not clip._load_torch(dev):
                out[dev] = {"error": "did not load"}
                continue
            clip.set_query("a test pattern")
            load_s = time.perf_counter() - t

            def embed(c=clip):
                from PIL import Image
                import cv2
                imgs = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
                embeds = []
                for i in range(0, len(imgs), 16):
                    o = c.infer(c._labels, imgs[i:i + 16])
                    embeds.append(o.image_embeds.float().cpu().numpy())
                return np.concatenate(embeds)

            vecs = embed()
            row = {"load_s": round(load_s, 2), "dtype": str(clip._dtype),
                   "held_mb": round(_settle() - before, 1),
                   **_timed(embed, len(frames))}
            if ref is None:
                ref = vecs
            else:
                row["cosine_vs_cpu"] = round(_row_cosines(vecs, ref), 5)
            out[dev] = row
            if dev == devices[-1]:
                out["app_picks"] = dev
                out["leak"] = _leak(embed)
            del clip
        except Exception as e:  # noqa: BLE001
            out[dev] = {"error": f"{type(e).__name__}: {e}"}
    return out


def _words(text: str) -> list:
    return [w.strip(".,!?;:").lower() for w in text.split() if w.strip(".,!?;:")]


def _word_agreement(a: str, b: str) -> float:
    """Share of words in common, in order (longest common subsequence)."""
    x, y = _words(a), _words(b)
    if not x and not y:
        return 1.0
    prev = [0] * (len(y) + 1)
    for wx in x:
        cur = [0]
        for j, wy in enumerate(y):
            cur.append(prev[j] + 1 if wx == wy else max(prev[j + 1], cur[j]))
        prev = cur
    return round(2 * prev[-1] / (len(x) + len(y)), 3)


def _whisper(video: str) -> dict:
    import whisper
    from modules.audio import transcript as tr
    from modules.system import mps_device

    audio = whisper.load_audio(video)
    out = {}
    for name in WHISPER_MODELS:
        res, ref = {}, None
        variants = [("cpu", "cpu", False)]
        if mps_device.device():
            variants += [("mps-fp32", "mps", False), ("mps-fp16", "mps", True)]
        for label, dev, fp16 in variants:
            before = _settle()
            try:
                t = time.perf_counter()
                model, got = tr.load_whisper(name, dev, print)
                load_s = time.perf_counter() - t
                params = dict(task="transcribe", temperature=0.0, beam_size=1,
                              best_of=1, language="en", verbose=None,
                              condition_on_previous_text=False, fp16=fp16)
                model.transcribe(audio, **params)       # warm
                t = time.perf_counter()
                text = model.transcribe(audio, **params)["text"].strip()
                row = {"device": got, "load_s": round(load_s, 2),
                       "transcribe_s": round(time.perf_counter() - t, 2),
                       "held_mb": round(_settle() - before, 1), "text": text[:200]}
                if ref is None:
                    ref = text
                else:
                    row["words_agree_vs_cpu"] = _word_agreement(text, ref)
                if label == "mps-fp32" and name == WHISPER_MODELS[0]:
                    row["leak"] = _leak(lambda m=model: m.transcribe(audio, **params), 4)
                res[label] = row
                del model
            except Exception as e:  # noqa: BLE001
                res[label] = {"error": f"{type(e).__name__}: {e}"}
        out[name] = res
    return out


def _object_detection(video: str) -> dict:
    """The app's own multi-process detection, at a few worker counts: wall time,
    the whole tree's peak memory, and the parent's memory after each run."""
    import tempfile
    import object_recognition
    from modules.system.smoke_test import _TreeWatch

    default = object_recognition.NUM_WORKERS
    out = {"default_workers": default}
    for workers in sorted({1, 2, default}):
        object_recognition.NUM_WORKERS = workers
        runs = []
        try:
            for _ in range(3):
                watch = _TreeWatch(max_tree_mb=1e9, max_children=10**6).start()
                t = time.perf_counter()
                object_recognition.run_object_detection(
                    video, ["person"], frame_skip=2,
                    csv_file=os.path.join(tempfile.mkdtemp(prefix="vh-bench-"), "o.csv"),
                    progress_fn=None, device="AUTO", log_fn=lambda *_: None)
                wall = time.perf_counter() - t
                watch.stop()
                runs.append({"wall_s": round(wall, 2),
                             "peak_tree_mb": round(watch.peak_tree_mb),
                             "peak_child_mb": round(watch.peak_child_mb),
                             "parent_after_mb": round(_settle(), 1)})
            out[f"workers_{workers}"] = runs
        except Exception as e:  # noqa: BLE001
            out[f"workers_{workers}"] = {"error": f"{type(e).__name__}: {e}", "runs": runs}
        finally:
            object_recognition.NUM_WORKERS = default
    return out


# ---------------------------------------------------------------------------

def main(argv) -> int:
    started = time.monotonic()
    video = _arg(argv, "--smoke-bench", "")
    report_path = _arg(argv, "--smoke-report", "")
    report = {"video": video, "frozen": bool(getattr(sys, "frozen", False)),
              "platform": sys.platform}
    if not video or not os.path.exists(video):
        print(f"❌ bench: no video at {video!r}")
        return 1

    def save():
        report["seconds"] = round(time.monotonic() - started, 1)
        if report_path:
            with open(report_path, "w", encoding="utf-8") as fh:
                json.dump(report, fh, indent=2)

    _section(report, "environment", _environment)
    try:
        from modules.system.device_utils import detect_best_device
        report["backend"] = detect_best_device(log_fn=print).backend_name
    except Exception as e:  # noqa: BLE001
        report["backend"] = f"error: {e}"
    frames = _frames(video)
    report["frames"] = len(frames)

    _section(report, "frame_encoder", lambda: _frame_encoder(frames))
    save()
    _section(report, "yolox", lambda: _yolox(frames))
    save()
    _section(report, "clip", lambda: _clip(frames))
    save()
    _section(report, "whisper", lambda: _whisper(video))
    save()
    _section(report, "object_detection", lambda: _object_detection(video))
    report["rss_end_mb"] = round(_settle(), 1)
    save()
    print(f"✅ bench finished in {report['seconds']}s")
    return 0
