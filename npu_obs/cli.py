"""npu_obs command line: probe the NPU, scan a recording, or watch OBS live.

    python -m npu_obs probe [--compare]
    python -m npu_obs video  RECORDING.mp4 [--every 0.5] [--classes person,dog]
    python -m npu_obs live   [--obs | --capture 1] [--chapters]

See npu_obs/README.md for what each does and what it costs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# --- shared ----------------------------------------------------------------

def _build_detector(args, prefer: str):
    from modules.vision.npu_detector import NPU, build_npu_detector
    fallback = () if args.npu_only else ("CPU",)
    det = build_npu_detector(args.model or "", prefer=prefer, device=args.device or NPU,
                             fallback=fallback, score_thr=args.score)
    print(f"🧠 Detector: {os.path.basename(det.model_path)} "
          f"on {det.device}, input {det.input_size[1]}x{det.input_size[0]}, "
          f"{len(det.class_names)} classes")
    return det


def _wanted(args, det) -> set[str] | None:
    if not args.classes:
        return None
    wanted = {c.strip() for c in args.classes.split(",") if c.strip()}
    unknown = wanted - set(det.class_names)
    if unknown:
        raise SystemExit(f"❌ Not classes of this model: {', '.join(sorted(unknown))}")
    return wanted


def _seen(dets, wanted: set[str] | None) -> dict[str, float]:
    """label -> best confidence among this frame's detections."""
    seen: dict[str, float] = {}
    for d in dets:
        if wanted is None or d.class_name in wanted:
            seen[d.class_name] = max(seen.get(d.class_name, 0.0), d.confidence)
    return seen


def _fmt_t(seconds: float) -> str:
    m, s = divmod(max(0.0, seconds), 60)
    h, m = divmod(int(m), 60)
    return f"{h}:{m:02d}:{s:05.2f}" if h else f"{m:02d}:{s:05.2f}"


def _yield_cpu(args) -> None:
    """Run below normal priority so the game, OBS and its encoder win any
    contest for the processor; the detector itself is on the NPU anyway."""
    if args.normal_priority:
        return
    try:
        import psutil
        proc = psutil.Process()
        proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
    except Exception as e:  # noqa: BLE001 - a nicety; never worth failing over
        print(f"[npu_obs] could not lower process priority: {e}")


def _add_detector_args(p: argparse.ArgumentParser, default_score: float) -> None:
    p.add_argument("--model", help="custom YOLOX model (.xml/.onnx); default: stock YOLOX")
    p.add_argument("--score", type=float, default=default_score,
                   help=f"minimum confidence (default {default_score})")
    p.add_argument("--classes", help="comma-separated class names to track (default: all)")
    p.add_argument("--device", help="OpenVINO device (default NPU)")
    p.add_argument("--npu-only", action="store_true",
                   help="fail instead of falling back to the CPU when the NPU can't run it")
    p.add_argument("--min-hits", type=int, default=2,
                   help="consecutive samples before a span opens (default 2)")
    p.add_argument("--normal-priority", action="store_true",
                   help="don't drop to below-normal process priority")


# --- probe -----------------------------------------------------------------

def cmd_probe(args) -> int:
    import numpy as np
    import openvino as ov
    from modules.vision.detection_backend import DEFAULT_MODEL_DIR
    from modules.vision.npu_detector import NpuYoloxDetector, npu_device

    core = ov.Core()
    print(f"OpenVINO {ov.__version__}")
    for d in core.available_devices:
        try:
            name = core.get_property(d, "FULL_DEVICE_NAME")
        except Exception:
            name = "?"
        print(f"  {d:7s} {name}")
    npu = npu_device(core)
    if not npu:
        print("\n❌ No NPU visible to OpenVINO. On a Core Ultra machine, install or "
              "update the Intel NPU driver.")
        if not args.compare:
            return 1

    models = sorted(Path(DEFAULT_MODEL_DIR).glob("*.xml"),
                    key=lambda p: p.with_suffix(".bin").stat().st_size if p.with_suffix(".bin").exists() else 0)
    if args.model:
        models = [Path(args.model)]
    if not models:
        print("\n❌ No YOLOX model — run: python tools/get_yolox_model.py nano tiny s")
        return 1
    devices = [npu] if npu else []
    if args.compare:
        devices += [d for d in core.available_devices if d != npu]

    from modules.vision.detection_backend import load_class_names
    names = load_class_names("yolo_objects_labels.json")
    frame = np.random.default_rng(0).integers(0, 255, (1080, 1920, 3), dtype=np.uint8)
    print(f"\n{'model':16s} {'device':7s} {'load':>7s} {'infer':>9s} "
          f"{'detect':>9s} {'CPU time/frame':>15s}")
    for xml in models:
        for dev in devices:
            t0 = time.perf_counter()
            try:
                det = NpuYoloxDetector(str(xml), names, device=dev, fallback=())
            except Exception as e:
                print(f"{xml.stem:16s} {dev:7s} ❌ {str(e).splitlines()[0][:70]}")
                continue
            load = time.perf_counter() - t0
            blob, _ = det._preprocess(frame)
            for _ in range(3):
                det._infer(blob)
            n = 100  # Windows' process clock ticks in 15.6 ms; average it out
            w0, c0 = time.perf_counter(), time.process_time()
            for _ in range(n):
                det._infer(blob)
            infer = (time.perf_counter() - w0) / n * 1000
            cpu = (time.process_time() - c0) / n * 1000
            w0 = time.perf_counter()
            for _ in range(10):
                det.detect(frame)
            full = (time.perf_counter() - w0) / 10 * 1000
            print(f"{xml.stem:16s} {det.device:7s} {load:6.2f}s {infer:7.2f}ms "
                  f"{full:7.2f}ms {cpu:13.2f}ms")
    print("\ninfer = model only; detect = 1080p frame incl. resize + decode + NMS (CPU);\n"
          "CPU time/frame = processor time this process spent per inference.")
    return 0


# --- video -----------------------------------------------------------------

def cmd_video(args) -> int:
    from npu_obs.sources import VideoFileSource, prefetch
    from npu_obs.spans import SpanTracker, summarize

    _yield_cpu(args)
    det = _build_detector(args, prefer=args.prefer)
    wanted = _wanted(args, det)
    src = VideoFileSource(args.video, every=args.every, hw_decode=not args.cpu_decode,
                          start=args.start, end=args.end)
    print(f"🎞️ {os.path.basename(args.video)}: {src.width}x{src.height} @ {src.fps:.2f} fps, "
          f"{_fmt_t(src.duration)}, sampling every {args.every}s "
          f"(~{src.samples} samples), decode: {'GPU' if src.hw_decode else 'CPU'}")

    max_gap = args.max_gap if args.max_gap is not None else max(1.5, 3 * args.every)
    tracker = SpanTracker(min_hits=args.min_hits, max_gap=max_gap)
    frames_out = open(args.frames, "w", encoding="utf-8") if args.frames else None
    n, det_s = 0, 0.0
    w0, c0 = time.perf_counter(), time.process_time()
    last_report = 0.0
    interrupted = False
    frames = prefetch(src)
    try:
        for t, frame in frames:
            d0 = time.perf_counter()
            dets = det.detect(frame)
            det_s += time.perf_counter() - d0
            n += 1
            tracker.update(t, _seen(dets, wanted))
            if frames_out:
                frames_out.write(json.dumps({"t": round(t, 3), "detections": [
                    {"label": d.class_name, "conf": round(d.confidence, 3),
                     "box": [round(d.x1), round(d.y1), round(d.x2), round(d.y2)]}
                    for d in dets if wanted is None or d.class_name in wanted]}) + "\n")
            now = time.perf_counter()
            if now - last_report > 2.0:
                last_report = now
                pct = f"{100 * n / src.samples:5.1f}%" if src.samples else ""
                print(f"\r  {pct} {_fmt_t(t)}  {n / (now - w0):5.1f} samples/s  "
                      f"open spans: {sum(1 for tr in tracker._tracks.values() if tr.open)}   ",
                      end="", flush=True)
    except KeyboardInterrupt:
        interrupted = True
        print("\n⏹️ Interrupted — writing what was scanned so far")
    finally:
        frames.close()  # stops the decoder thread before the capture goes
        src.close()
        if frames_out:
            frames_out.close()
    tracker.finish()
    wall = time.perf_counter() - w0
    cpu = time.process_time() - c0
    spans = sorted(tracker.closed, key=lambda s: s.start)

    out = Path(args.out) if args.out else Path(f"{Path(args.video).stem}.npu.json")
    result = {
        "video": os.path.abspath(args.video),
        "detector": {"model": args.model or "stock YOLOX", "device": det.device,
                     "score": args.score, "classes": sorted(wanted) if wanted else None},
        "sampling": {"every": args.every, "samples": n, "min_hits": args.min_hits,
                     "max_gap": max_gap, "complete": not interrupted},
        "summary": summarize(spans),
        "spans": [s.to_dict() for s in spans],
    }
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"\r✅ {n} samples in {wall:.1f}s ({n / wall if wall else 0:.1f}/s); "
          f"detector {det_s / max(n, 1) * 1000:.1f} ms/sample on {det.device}; "
          f"process CPU {cpu / wall * 100 if wall else 0:.0f}% of one core")
    if spans:
        print(f"\n{'label':20s} {'spans':>6s} {'on screen':>10s} {'peak':>6s}")
        for label, row in result["summary"].items():
            print(f"{label:20s} {row['spans']:6d} {_fmt_t(row['seconds']):>10s} {row['peak']:6.2f}")
    else:
        print("No spans found (lower --score or --min-hits to see more).")
    print(f"\n📄 {out}")
    return 0


# --- live ------------------------------------------------------------------

def cmd_live(args) -> int:
    from npu_obs.spans import SpanTracker

    _yield_cpu(args)
    det = _build_detector(args, prefer=args.prefer)
    wanted = _wanted(args, det)
    client = None
    if args.capture is not None:
        from npu_obs.sources import CaptureSource
        device = int(args.capture) if str(args.capture).isdigit() else args.capture
        src = CaptureSource(device)
        print(f"📷 Capture device {device!r}")
    else:
        from npu_obs.obs_ws import ObsClient, ObsError
        from npu_obs.sources import ObsScreenshotSource
        password = args.obs_password or os.environ.get("OBS_WEBSOCKET_PASSWORD", "")
        try:
            client = ObsClient(args.obs_host, args.obs_port, password).connect()
            version = client.request("GetVersion")
            src = ObsScreenshotSource(client, args.obs_source,
                                      width=max(det.input_size))
        except ObsError as e:
            print(f"❌ {e}")
            return 1
        print(f"🎬 OBS {version.get('obsVersion', '?')} — watching "
              f"{'scene/source ' + repr(src.source) if args.obs_source else 'the program scene'} "
              f"at {src.width}px wide")

    out = Path(args.out) if args.out else Path(time.strftime("npu_live_%Y%m%d_%H%M%S.jsonl"))
    log = open(out, "a", encoding="utf-8")
    tracker = SpanTracker(min_hits=args.min_hits,
                          max_gap=args.max_gap if args.max_gap is not None
                          else max(1.5, 4 * args.interval))
    chapters = args.chapters and client is not None
    chapter_warned = False
    print(f"👀 Sampling every {args.interval}s — Ctrl+C to stop. Events → {out}")

    n, det_s, w0, c0 = 0, 0.0, time.perf_counter(), time.process_time()
    stat_n, stat_det, stat_w, stat_c = 0, 0.0, w0, c0
    last_t, last_rec = 0.0, None   # so spans still open at exit get a position too

    def emit(kind, span, t, rec_ms):
        nonlocal chapter_warned
        rec_at = None
        if rec_ms is not None:
            rec_at = rec_ms - int(((t - span.start) if kind == "start" else (t - span.end)) * 1000)
        row = {"event": kind, "wall": time.strftime("%Y-%m-%dT%H:%M:%S"),
               **span.to_dict(), "record_ms": rec_at}
        log.write(json.dumps(row) + "\n")
        log.flush()
        rec = f"  rec {_fmt_t(rec_at / 1000)}" if rec_at is not None else ""
        icon = "🟢" if kind == "start" else "⚪"
        print(f"\n{icon} {kind:5s} {span.label:18s} peak {span.peak:.2f}{rec}", end="")
        if chapters and kind == "start" and rec_ms is not None:
            try:
                client.request("CreateRecordChapter", {"chapterName": span.label})
            except Exception as e:  # noqa: BLE001 - old OBS / not hybrid MP4
                if not chapter_warned:
                    chapter_warned = True
                    print(f"\n⚠️ Could not add a recording chapter ({e}). Chapters need "
                          "OBS 30.2+ recording to Hybrid MP4; events are still logged.")

    try:
        while True:
            tick = time.perf_counter()
            t, frame = src.read()
            rec_ms = src.record_ms() if client is not None else None
            last_t, last_rec = t, rec_ms
            d0 = time.perf_counter()
            dets = det.detect(frame)
            dt = time.perf_counter() - d0
            det_s += dt
            n += 1
            for kind, span in tracker.update(t, _seen(dets, wanted)):
                emit(kind, span, t, rec_ms)
            if args.preview:
                _show_preview(frame, dets, wanted, f"{det.device}  {os.path.basename(det.model_path)}  "
                              f"{dt * 1000:.1f} ms  (q to quit)")
            now = time.perf_counter()
            if now - stat_w >= 10:
                k = n - stat_n
                print(f"\n  … {k / (now - stat_w):4.1f} samples/s, detector "
                      f"{(det_s - stat_det) / max(k, 1) * 1000:.1f} ms on {det.device}, "
                      f"CPU {(time.process_time() - stat_c) / (now - stat_w) * 100:.0f}% "
                      f"of one core", end="")
                stat_n, stat_det, stat_w, stat_c = n, det_s, now, time.process_time()
            time.sleep(max(0.0, args.interval - (time.perf_counter() - tick)))
    except KeyboardInterrupt:
        pass
    except Exception as e:  # noqa: BLE001 - OBS closed, device unplugged, …
        print(f"\n❌ {e}")
    finally:
        t_end = time.perf_counter() - w0
        for kind, span in tracker.finish():
            emit(kind, span, last_t, last_rec)
        log.close()
        src.close()
        if client is not None:
            client.close()
        if args.preview:
            import cv2
            cv2.destroyAllWindows()
    print(f"\n⏹️ {n} samples in {t_end:.0f}s; events in {out}")
    return 0


PREVIEW_WINDOW = "npu_obs preview"


def _show_preview(frame, dets, wanted, status: str = "") -> None:
    """Draw the boxes and a status line; raises KeyboardInterrupt when the user
    presses q / Esc or closes the window, which ends the run like Ctrl+C."""
    import cv2
    view = frame.copy()
    for d in dets:
        if wanted is not None and d.class_name not in wanted:
            continue
        p1, p2 = (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2))
        cv2.rectangle(view, p1, p2, (0, 200, 255), 2)
        cv2.putText(view, f"{d.class_name} {d.confidence:.2f}", (p1[0], max(12, p1[1] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)
    if status:
        cv2.rectangle(view, (0, 0), (view.shape[1], 24), (0, 0, 0), -1)
        cv2.putText(view, status, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow(PREVIEW_WINDOW, view)
    key = cv2.waitKey(1) & 0xFF
    closed = cv2.getWindowProperty(PREVIEW_WINDOW, cv2.WND_PROP_VISIBLE) < 1
    if key in (ord("q"), 27) or closed:
        raise KeyboardInterrupt


# --- entry -----------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m npu_obs", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("probe", help="check the NPU and time the detector on it")
    p.add_argument("--compare", action="store_true", help="also time CPU and GPU devices")
    p.add_argument("--model", help="time this model instead of the installed stock ones")
    p.set_defaults(func=cmd_probe)

    p = sub.add_parser("video", help="scan a recording for spans where classes appear")
    p.add_argument("video")
    _add_detector_args(p, default_score=0.35)
    p.add_argument("--prefer", default="large", choices=("small", "large"),
                   help="stock model size: most accurate installed (default) or fastest")
    p.add_argument("--every", type=float, default=0.5, help="seconds between samples")
    p.add_argument("--start", type=float, default=0.0, help="start at this second")
    p.add_argument("--end", type=float, help="stop at this second")
    p.add_argument("--max-gap", type=float,
                   help="seconds a class may vanish before its span ends (default max(1.5, 3x every))")
    p.add_argument("--cpu-decode", action="store_true",
                   help="decode on the CPU instead of the GPU (about twice the CPU load)")
    p.add_argument("--out", help="result JSON (default <video name>.npu.json here)")
    p.add_argument("--frames", help="also write every sample's detections to this JSONL")
    p.set_defaults(func=cmd_video)

    p = sub.add_parser("live", help="watch OBS (or a capture device) and log spans as they happen")
    _add_detector_args(p, default_score=0.35)
    p.add_argument("--prefer", default="small", choices=("small", "large"),
                   help="stock model size: fastest installed (default) or most accurate")
    p.add_argument("--interval", type=float, default=0.25, help="seconds between samples")
    p.add_argument("--max-gap", type=float,
                   help="seconds a class may vanish before its span ends (default max(1.5, 4x interval))")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--obs", action="store_true", help="read from OBS via obs-websocket (default)")
    src.add_argument("--capture", help="capture device index (e.g. OBS Virtual Camera) or stream URL")
    p.add_argument("--obs-host", default="localhost")
    p.add_argument("--obs-port", type=int, default=4455)
    p.add_argument("--obs-password", help="or set OBS_WEBSOCKET_PASSWORD")
    p.add_argument("--obs-source", default="", help="scene/source to watch (default: program scene)")
    p.add_argument("--chapters", action="store_true",
                   help="add an OBS recording chapter when a span starts (OBS 30.2+, Hybrid MP4)")
    p.add_argument("--preview", action="store_true", help="show a window with the boxes")
    p.add_argument("--out", help="events JSONL (default npu_live_<time>.jsonl here)")
    p.set_defaults(func=cmd_live)
    return ap


def main(argv: list[str] | None = None) -> int:
    from modules.system.debug_console import force_utf8_stdio
    force_utf8_stdio()
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"❌ {e}")
        return 1
