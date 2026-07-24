"""On-demand single-analysis runners for the timeline viewer.

The timeline viewer is otherwise read-only over whatever the main GUI already
computed. When a cache is missing actions, objects, or a transcript, the user
had to go back to the first GUI and re-run the whole pipeline. This module lets
the viewer run *one* analysis on the loaded video and fold the result straight
into the cache.

Design:
- Thin wrappers over the same entry points the pipeline uses
  (`get_transcript_segments`, `run_action_detection`, `run_object_detection_single`
  fed an ultralytics YOLO model), so behaviour and model choice never drift
  from a full run.
- The *advanced* knobs stay in the first GUI. Here we read that GUI's saved
  `config.yaml` for defaults (Whisper model, object list, confidence, sample
  rate, YOLO model size), so a viewer button runs "what the main GUI is
  currently set to".
- Every heavy import (torch, whisper, ultralytics) lives inside the function
  that needs it, so importing this module stays cheap and the viewer never
  hard-depends on a model runtime at construction time.
- Each runner takes a uniform `progress(current, total, task, details)`
  callback — the shape all three underlying functions already emit — and an
  optional `cancel` `threading.Event`. Each returns data already in the
  on-disk cache shape (see `pipeline.collect_analysis_data`), so the caller
  only has to merge and redraw.
"""
from __future__ import annotations

import os
from typing import Callable, Optional

ProgressFn = Optional[Callable[[int, int, str, str], None]]


def analysis_defaults() -> dict:
    """Read the main GUI's config.yaml for the settings these runs need.

    Best-effort: a missing file or key falls back to the same defaults the
    pipeline uses, so the viewer still works on a fresh install.
    """
    cfg = {}
    try:
        from modules.app_paths import config_path
        path = config_path("config.yaml")
    except Exception:
        path = "config.yaml"
    try:
        if os.path.exists(path):
            import yaml
            with open(path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
    except Exception:
        cfg = {}

    objects_cfg = cfg.get("objects", {}) or {}
    transcript_cfg = cfg.get("transcript", {}) or {}
    advanced_cfg = cfg.get("advanced", {}) or {}

    return {
        "action_list": list((cfg.get("actions", {}) or {}).get("interesting", []) or []),
        "object_list": list(objects_cfg.get("interesting", []) or []),
        # config stores confidence as a 0-100 int (30 → 0.30)
        "object_confidence": float(objects_cfg.get("confidence", 30)) / 100.0,
        "object_frame_skip": int(advanced_cfg.get("object_frame_skip", 10) or 10),
        "yolo_model_size": str(advanced_cfg.get("yolo_model_size", "n") or "n"),
        "yolo_type": advanced_cfg.get("yolo_type", "standard") or "standard",
        "whisper_model": transcript_cfg.get("model", "base") or "base",
        "language": transcript_cfg.get("source_lang", "en") or "en",
        "transcript_enabled": bool(transcript_cfg.get("enabled", False)),
        "search_keywords": list(transcript_cfg.get("search_keywords", []) or []),
        "sample_rate": int(advanced_cfg.get("sample_rate", 5) or 5),
    }


# --------------------------------------------------------------------------- #
# Transcript
# --------------------------------------------------------------------------- #
def run_transcript(video_path: str, *, model: Optional[str] = None,
                   language: Optional[str] = None, progress: ProgressFn = None,
                   cancel=None, log=print) -> dict:
    """Transcribe the video and return a cache-shaped `transcript` dict.

    Also writes the sibling `<video>_transcript.txt` the Transcript panel reads,
    matching the pipeline's behaviour.
    """
    from modules.transcript import get_transcript_segments
    d = analysis_defaults()
    model = model or d["whisper_model"]
    language = language or d["language"]

    segments = get_transcript_segments(
        video_path, model_name=model, progress_fn=progress, log_fn=log,
        language=language, enable_diarization=False,
    )
    if cancel is not None and cancel.is_set():
        raise _Cancelled()

    # Persist the plain-text sibling so the Transcript tab finds it on reopen.
    try:
        from modules.transcript_srt import create_enhanced_transcript
        base = os.path.splitext(video_path)[0]
        with open(f"{base}_transcript.txt", "w", encoding="utf-8") as f:
            f.write(create_enhanced_transcript(segments))
    except Exception as e:
        log(f"⚠️ Could not write transcript sidecar: {e}")

    return {
        "segments": segments or [],
        "language": language,
        "cached_full_transcript": True,
        "keyword_filtered": False,
    }


# --------------------------------------------------------------------------- #
# Actions
# --------------------------------------------------------------------------- #
def _actions_to_cache(dets) -> list:
    """Normalize raw action detections to the on-disk cache shape. Mirrors
    pipeline.collect_analysis_data._actions_to_cache, tolerating 4/5/6-tuples."""
    out = []
    for det in dets or []:
        if len(det) == 6:
            ts, frame_id, action_id, score, name, _model = det
        elif len(det) == 5:
            ts, frame_id, action_id, score, name = det
        elif len(det) == 4:
            ts, frame_id, score, name = det
            action_id = -1
        else:
            continue
        out.append({
            "timestamp": float(ts),
            "frame_id": int(frame_id),
            "action_id": int(action_id),
            "confidence": float(score),
            "action_name": str(name),
        })
    return out


def run_actions(video_path: str, *, sample_rate: Optional[int] = None,
                interesting_actions: Optional[list] = None,
                progress: ProgressFn = None, cancel=None, log=print) -> list:
    """Run action recognition and return the cache-shaped `actions` list
    (every detection — the timeline's "show all" source).

    `interesting_actions` is an optional keep-list: blank/None detects and
    keeps all actions; a list narrows the result to those names (same filter
    the pipeline's `interesting_actions` applies)."""
    from action_recognition import run_action_detection
    d = analysis_defaults()
    sample_rate = sample_rate or d["sample_rate"]
    keep = [a.strip() for a in (interesting_actions or []) if a and a.strip()] or None

    detections, _bboxes = run_action_detection(
        video_path=video_path,
        sample_rate=sample_rate,
        interesting_actions=keep,
        progress_callback=progress,
        cancel_flag=cancel,
        draw_bboxes=False,
        use_person_detection=True,
        include_model_type=False,
    )
    if cancel is not None and cancel.is_set():
        raise _Cancelled()
    return _actions_to_cache(detections)


# --------------------------------------------------------------------------- #
# Objects
# --------------------------------------------------------------------------- #
def _load_yolo(d: dict, log=print):
    """Load an ultralytics YOLO detector the way the pipeline does: prefer a
    pre-exported OpenVINO folder for the chosen size, else the .pt (ultralytics
    fetches the weights if they aren't present)."""
    from ultralytics import YOLO
    size = d["yolo_model_size"]
    ov_folder = f"yolo11{size}_openvino_model/"
    pt_path = f"yolo11{size}.pt"
    if os.path.isdir(ov_folder):
        log(f"✅ Object detector: YOLO OpenVINO ({ov_folder})")
        return YOLO(ov_folder, task="detect")
    log(f"✅ Object detector: YOLO {pt_path}")
    return YOLO(pt_path)


def run_objects(video_path: str, objects: list, *, progress: ProgressFn = None,
                cancel=None, log=print) -> list:
    """Run object detection for the given class list and return the
    cache-shaped `objects` list `[{timestamp, objects, count}, ...]`.

    Raises ValueError if no classes were given — object detection has nothing
    to look for without a list.
    """
    objects = [o.strip() for o in (objects or []) if o and o.strip()]
    if not objects:
        raise ValueError("No object classes given — type at least one (e.g. person, car).")

    from object_recognition import run_object_detection_single
    d = analysis_defaults()

    model = _load_yolo(d, log)
    if model is None:
        raise RuntimeError("Object detector unavailable — could not load a YOLO model.")

    det_by_sec, _bboxes = run_object_detection_single(
        video_path, model, objects,
        log_fn=log, progress_fn=progress,
        frame_skip=d["object_frame_skip"],
        cancel_flag=cancel, draw_boxes=False,
        confidence_threshold=d["object_confidence"],
    )
    if cancel is not None and cancel.is_set():
        raise _Cancelled()

    return [
        {"timestamp": int(sec), "objects": [str(o) for o in objs], "count": len(objs)}
        for sec, objs in sorted(det_by_sec.items())
    ]


class _Cancelled(Exception):
    """Raised when a cancel Event was set mid-run; the caller treats it as a
    quiet abort rather than an error."""
