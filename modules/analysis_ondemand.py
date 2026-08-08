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
        "frame_skip": int(advanced_cfg.get("frame_skip", 5) or 5),
    }


# --------------------------------------------------------------------------- #
# Transcript
# --------------------------------------------------------------------------- #
def _write_transcript_sidecar(video_path: str, segments: list, *,
                              only_if_missing: bool = False, log=print) -> None:
    """Write the plain-text `<video>_transcript.txt` the Transcript tab reads."""
    try:
        base = os.path.splitext(video_path)[0]
        path = f"{base}_transcript.txt"
        if only_if_missing and os.path.exists(path):
            return
        from modules.transcript_srt import create_enhanced_transcript
        with open(path, "w", encoding="utf-8") as f:
            f.write(create_enhanced_transcript(segments))
    except Exception as e:
        log(f"⚠️ Could not write transcript sidecar: {e}")


def cached_transcript(video_path: str, *, language: Optional[str] = None,
                      log=print) -> Optional[dict]:
    """The transcript already in this video's cache, when it can stand in for a
    fresh one — otherwise ``None``.

    Transcription is the most expensive thing either surface can start, and some
    of what the buttons offer is *derived* from a transcript rather than being
    one: subtitles are an `.srt` written from segments, which the cache may
    already hold. Reuse is only sound when the cached transcript covers the
    whole video — a keyword-filtered one kept only the moments around the search
    words, so subtitles built from it would be full of holes — and when it is in
    the language this run asked for.

    Looks in every cache file the video owns, not just the newest: a video here
    had a 1362-segment transcript in one file and an empty one in its sibling.
    """
    wanted = language or analysis_defaults()["language"]
    best, rejected = None, []

    for data in read_caches(video_path):
        tr = (data or {}).get("transcript") or {}
        segments = tr.get("segments") or []
        if not segments:
            continue
        if tr.get("keyword_filtered"):
            rejected.append("ℹ️ The cached transcript covers only the keyword "
                            "matches — transcribing the whole video instead")
            continue
        cached_lang = tr.get("language")
        if wanted and wanted != "auto" and cached_lang and cached_lang != wanted:
            rejected.append(f"ℹ️ The cached transcript is '{cached_lang}' but this "
                            f"run asked for '{wanted}' — transcribing again")
            continue
        if best is None or len(segments) > len(best.get("segments") or []):
            best = tr

    if best is not None:
        return best
    if rejected:
        log(rejected[0])   # why the transcript that *is* there cannot be used
    return None


def run_transcript(video_path: str, *, model: Optional[str] = None,
                   language: Optional[str] = None, progress: ProgressFn = None,
                   cancel=None, log=print) -> dict:
    """Transcribe the video and return a cache-shaped `transcript` dict.

    Also writes the sibling `<video>_transcript.txt` the Transcript panel reads,
    matching the pipeline's behaviour.

    Always transcribes: this is the button whose whole purpose is producing a
    transcript, so pressing it with one already cached means "do it again" (a
    different model or language, or a cache you no longer trust). Runs that only
    *need* a transcript should ask `cached_transcript` first.
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
    _write_transcript_sidecar(video_path, segments, log=log)

    return {
        "segments": segments or [],
        "language": language,
        "cached_full_transcript": True,
        "keyword_filtered": False,
    }


# --------------------------------------------------------------------------- #
# Subtitles
# --------------------------------------------------------------------------- #
def _band(progress: ProgressFn, lo: int, hi: int) -> ProgressFn:
    """Squeeze a sub-step's own 0-100 into ``[lo, hi]`` of the caller's bar.

    A run made of two long passes (transcribe, then translate) would otherwise
    have each of them drive the bar from 0 to 100 in turn, which reads as the
    work restarting.
    """
    if progress is None:
        return None

    def report(current, total, task="", details=""):
        try:
            frac = float(current) / float(total) if total else 0.0
        except Exception:
            frac = 0.0
        frac = max(0.0, min(1.0, frac))
        progress(int(lo + frac * (hi - lo)), 100, task, details)

    return report


def run_subtitles(video_path: str, *, model: Optional[str] = None,
                  language: Optional[str] = None, source_lang: Optional[str] = None,
                  target_lang: Optional[str] = None, reuse_cached: bool = True,
                  progress: ProgressFn = None, cancel=None, log=print) -> dict:
    """Write a full-video `.srt` next to the video, then return the same
    cache-shaped `transcript` dict `run_transcript` returns.

    Thin wrapper over `run_transcript` (so it also writes the `_transcript.txt`
    sidecar and folds identically into the cache) plus `create_srt_file`. When
    `target_lang` differs from the spoken language the subtitles are translated;
    the file is named `<video>_<lang>.srt` with the language actually written.

    Transcribes only when it has to. What this produces is the `.srt`, and a
    transcript of that video may already be sitting in the cache from an earlier
    run — re-deriving it costs minutes to hours for a file that takes seconds to
    write. `reuse_cached=False` forces a fresh pass.

    The spoken language is read off the transcript that ends up being used, so a
    reused one is labelled with *its* language rather than with what this run
    happened to ask for. `source_lang` is only a fallback for a transcript that
    does not record one.
    """
    # Transcribing owns the first stretch of the bar and translating the rest;
    # a reused transcript simply starts at the boundary.
    TRANSCRIBE_TO = 60

    tr = cached_transcript(video_path, language=language, log=log) if reuse_cached else None
    if tr is not None:
        n = len(tr.get("segments") or [])
        log(f"✅ Using the transcript already in this video's cache ({n} segments) "
            f"— no re-transcription")
        if progress:
            progress(TRANSCRIBE_TO, 100, "Subtitles",
                     f"Reusing cached transcript ({n} segments)")
        # An older cache can predate the sidecar the Transcript tab reads, and
        # the run that would have written it is the one being skipped.
        _write_transcript_sidecar(video_path, tr.get("segments") or [],
                                  only_if_missing=True, log=log)
    else:
        tr = run_transcript(video_path, model=model, language=language,
                            progress=_band(progress, 0, TRANSCRIBE_TO),
                            cancel=cancel, log=log)
    segments = tr.get("segments", []) or []

    from modules.transcript_srt import create_srt_file
    base = os.path.splitext(video_path)[0]
    src = tr.get("language") or language or source_lang or "en"
    translating = bool(target_lang and target_lang != src)
    out_lang = target_lang if translating else src
    # "auto" is a request to Whisper, not a language: a file called
    # `movie_auto.srt` names nothing. Untranslated, it is just the subtitles.
    srt_path = f"{base}.srt" if out_lang == "auto" else f"{base}_{out_lang}.srt"

    if cancel is not None and cancel.is_set():
        raise _Cancelled()

    if progress:
        progress(TRANSCRIBE_TO, 100, "Subtitles",
                 f"Translating {len(segments)} segments to {target_lang}..." if translating
                 else f"Writing {os.path.basename(srt_path)}...")

    # create_srt_file translates internally when target_lang != source_lang —
    # hundreds of LLM batches for a long video, and the second place a subtitle
    # run can sit for minutes, so it gets the rest of the bar.
    create_srt_file(segments, srt_path, source_lang=src,
                    target_lang=target_lang if translating else None,
                    progress_fn=_band(progress, TRANSCRIBE_TO, 98))
    log(f"✅ Subtitles saved: {srt_path}")
    return tr


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


# --------------------------------------------------------------------------- #
# Motion & scenes
# --------------------------------------------------------------------------- #
def run_motion(video_path: str, *, frame_skip: Optional[int] = None,
               progress: ProgressFn = None, cancel=None, log=print) -> dict:
    """Run scene-cut + motion detection and return a cache-shaped patch with
    `scenes`, `motion_events` and `motion_peaks`.

    One detector pass produces all three signals — which is why the three
    scoring rows share a single button. Runs on CPU (the universally-available
    device; the offline pipeline falls back to it too when there's no CUDA).
    """
    from modules.motion_scene_detect_optimized import detect_scenes_motion_optimized
    d = analysis_defaults()
    fs = frame_skip if frame_skip is not None else d["frame_skip"]

    result = detect_scenes_motion_optimized(
        video_path, frame_skip=int(fs), device="cpu",
        cancel_flag=cancel, progress_callback=progress,
    )
    if cancel is not None and cancel.is_set():
        raise _Cancelled()
    scenes, motion_events, motion_peaks = result if (result and len(result) == 3) else ([], [], [])
    return {
        "scenes": [{"start": float(s), "end": float(e)} for s, e in scenes],
        "motion_events": [float(t) for t in motion_events],
        "motion_peaks": [float(t) for t in motion_peaks],
    }


# --------------------------------------------------------------------------- #
# Audio peaks
# --------------------------------------------------------------------------- #
def run_audio(video_path: str, *, progress: ProgressFn = None,
              cancel=None, log=print) -> dict:
    """Run audio-peak detection (plus the waveform the timeline viewer draws)
    and return a cache-shaped patch under both the modern `audio` block and the
    legacy `audio_peaks` key."""
    from modules.audio_peaks import extract_audio_peaks, extract_waveform_data
    if progress:
        progress(0, 1, "Audio", "Detecting audio peaks…")
    peaks = [float(t) for t in (extract_audio_peaks(video_path, cancel_flag=cancel) or [])]
    if cancel is not None and cancel.is_set():
        raise _Cancelled()
    try:
        waveform = extract_waveform_data(video_path)
    except Exception as e:
        log(f"⚠️ Waveform extraction failed: {e}")
        waveform = None
    if progress:
        progress(1, 1, "Audio", f"{len(peaks)} peaks")
    return {
        "audio_peaks": peaks,
        "audio": {"peaks": peaks, "waveform": waveform},
    }


# --------------------------------------------------------------------------- #
# Cache fold
# --------------------------------------------------------------------------- #
def _cache_files(video_path: str):
    """``(video_hash, [cache files])`` for a video — the shared discovery rule.

    Extracted so a reader and a writer can never disagree about which files
    belong to a video: the timeline viewer picks its file signature-first (then
    newest) rather than by glob order, so anything touching one file has to know
    about all of them.
    """
    from pathlib import Path
    from modules.video_cache import VideoAnalysisCache

    cache_dir = Path("./cache")
    cache_dir.mkdir(exist_ok=True)
    video_hash = VideoAnalysisCache()._get_video_hash(video_path)
    if not video_hash:
        return "", []
    return video_hash, sorted(cache_dir.glob(f"{video_hash}*.cache.json"))


def read_caches(video_path: str) -> list:
    """Every readable on-disk cache for a video, newest first.

    One video can own several cache files — the pipeline names them by
    parameter signature, and older runs leave a legacy `<hash>.cache.json`
    behind. `merge_into_cache` deliberately writes a result into *all* of them,
    so a reader hunting for one signal has to look in all of them too: the
    newest file is not necessarily the one holding the transcript.
    """
    import json

    out = []
    _hash, files = _cache_files(video_path)
    for path in sorted(files, key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            with open(path, "r", encoding="utf-8") as f:
                out.append(json.load(f))
        except Exception:
            continue
    return out


def read_cache(video_path: str) -> dict:
    """The newest on-disk cache for a video, or ``{}`` when there is none."""
    caches = read_caches(video_path)
    return caches[0] if caches else {}


def run_composition(video_path: str, *, log=print) -> dict:
    """Re-apply ``composition_rules.yaml`` to boxes already in the cache.

    The engine is pure: it reads per-frame boxes and writes event names, and
    never touches the video or a detector. But in ``pipeline.py`` it only runs
    *inside* a fresh object-detection pass, so editing one rule meant
    re-detecting the whole video to see the effect — minutes or hours for a
    change that takes the engine milliseconds. This runs it on its own.

    Returns a cache patch containing the rebuilt ``objects`` list.

    Idempotent by construction: every name the current rule set can emit is
    stripped from the cached seconds before the new results are folded in, so
    running it repeatedly converges instead of accumulating duplicates. Because
    the strip list comes from the rules file, deleting a rule also removes its
    events on the next run.

    Composed boxes are deliberately NOT written back to ``object_bboxes``. That
    list is the detector's record; keeping derived entries out of it is what
    makes re-running safe, since a composed entry is otherwise
    indistinguishable from a detection.
    """
    from modules.app_paths import composition_rules_path
    from video_ai_editor.composition_engine import CompositionEngine

    rules_path = composition_rules_path()
    if not rules_path:
        raise RuntimeError(
            "No composition_rules.yaml found — add rules in the Advanced tab first.")

    cache = read_cache(video_path)
    if not cache:
        raise RuntimeError("No analysis cache for this video — run an analysis first.")

    bboxes = cache.get("object_bboxes") or []
    if not bboxes:
        raise RuntimeError(
            "No object boxes in the cache — run object detection first "
            "(the rules match against its boxes).")

    engine = CompositionEngine(rules_path)
    known = set(engine.event_names)
    if not known:
        raise RuntimeError(f"No events defined in {rules_path}.")

    composed, _composed_bb = engine.run(bboxes)

    # Rebuild the per-second list: drop this rule set's previous output, keep
    # every real detection, then fold in the new events.
    by_sec: dict = {}
    for entry in cache.get("objects") or []:
        sec = int(entry.get("timestamp", 0))
        names = [n for n in (entry.get("objects") or []) if n not in known]
        if names:
            by_sec[sec] = names
    for sec, names in composed.items():
        merged = set(by_sec.get(int(sec), [])) | set(names)
        by_sec[int(sec)] = sorted(merged)

    rebuilt = [
        {"timestamp": sec, "objects": names, "count": len(names)}
        for sec, names in sorted(by_sec.items())
    ]
    hits = sum(len(v) for v in composed.values())
    log(f"🧩 Composition: {hits} event-hit(s) over {len(composed)} second(s) "
        f"from {len(bboxes)} cached frames")
    # Also record WHICH names are derived, so the timeline can group them apart
    # from real detections. Written on every run so the list tracks the current
    # rules — a renamed or deleted rule stops being claimed as an event.
    return {"objects": rebuilt, "composed_event_names": sorted(known)}


def merge_into_cache(video_path: str, patch: dict, *, seed: dict = None,
                     log=print) -> bool:
    """Fold one signal's result into a video's on-disk cache, leaving the other
    signals intact.

    The single fold path for every on-demand run, from either surface (the main
    window's per-signal buttons and the timeline viewer's Analyze panel): a
    per-signal run must *merge*, never overwrite the whole entry (the full
    pipeline writes every signal at once, so driving a single-signal run through
    it would clobber the rest).

    When the video has no cache yet, seeds a legacy `<hash>.cache.json`. Pass
    `seed` (e.g. the caller's in-memory cache_data) to carry any already-known
    signals into that fresh file rather than starting from just this patch.
    Best-effort.
    """
    try:
        import json
        from pathlib import Path
        from modules.video_cache import atomic_write_json

        cache_dir = Path("./cache")
        video_hash, matching = _cache_files(video_path)
        if not video_hash:
            log("⚠️ No video_hash; analysis not persisted")
            return False

        # Every write goes through atomic_write_json (tmp file + replace) rather
        # than open(..., "w"). A plain open truncates the destination before the
        # new content is written, so anything that stops the process mid-write --
        # a cancel, a crash, power loss, the OS killing a long run -- leaves a
        # half-written file where a complete cache used to be. Observed for real:
        # a run killed during this write truncated a cache mid-array and cost the
        # object_bboxes for a 60-minute video, which is expensive to regenerate.
        # A cache is derived data, but it is derived over hours.
        if matching:
            # Fold into EVERY matching cache file, not just the first: the
            # timeline viewer picks its file signature-first (then newest), which
            # isn't necessarily glob order, so updating only one can leave the
            # signal in a file the viewer never reads.
            wrote = 0
            for cache_file in matching:
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        disk = json.load(f)
                except Exception:
                    continue
                disk.update(patch)
                atomic_write_json(Path(cache_file), disk)
                wrote += 1
            log(f"💾 Analysis merged → {wrote} cache file(s)")
            return wrote > 0
        else:
            cache_file = cache_dir / f"{video_hash}.cache.json"
            disk = dict(seed) if seed else {}
            disk.setdefault("video_path", str(video_path))
            disk["video_hash"] = video_hash
            disk["cache_complete"] = True
            disk.update(patch)
            atomic_write_json(cache_file, disk)
            log(f"💾 Analysis merged → {cache_file.name}")
            return True
    except Exception as e:
        log(f"⚠️ Could not persist analysis: {e}")
        return False


class _Cancelled(Exception):
    """Raised when a cancel Event was set mid-run; the caller treats it as a
    quiet abort rather than an error."""
