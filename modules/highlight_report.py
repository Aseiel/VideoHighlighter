"""Explain why each highlight was chosen — as data, then as a page.

The pipeline already computes a full justification for every moment it keeps:
per-signal point breakdown, which objects and actions fired, the confidence
tier an action landed in, and whether the multi-signal boost applied. Until now
that went to ``print()`` and was discarded with the debug log.

This module turns the same numbers into a structured record and renders it.
``build_report`` does the attribution, and the renderers are views over its
output — so the text a developer reads in the debug log and the page a user
opens can never disagree about what happened.

Three constraints shaped it:

* **No new dependencies.** ``matplotlib`` is ``--exclude-module``'d from the
  frozen build, so score bars are CSS. ``numpy`` is already a hard dependency
  and is the only import beyond the standard library.
* **Self-contained output.** The HTML embeds its styles and its thumbnails as
  ``data:`` URIs, so the report is one file a user can email. Nothing is
  fetched when it is opened.
* **No knowledge of editions.** Signals arrive as a dict; whichever ones the
  caller ran are the ones reported. A build with extra detectors produces a
  richer report through the same code path, with nothing to gate.

Thumbnails are injected rather than extracted here (``thumbnail_fn``), which
keeps the module testable without a video file or OpenCV.
"""
from __future__ import annotations

import base64
import datetime as _dt
import html
import json
from typing import Callable, Iterable, Mapping, Optional, Sequence

import numpy as np


# Order matters: it is the order signals appear in the breakdown, chosen so the
# cheap ambient signals read first and the semantic ones last.
SIGNAL_LABELS = (
    ("scene", "Scene change"),
    ("motion_event", "Motion event"),
    ("motion_peak", "Motion peak"),
    ("audio", "Audio peak"),
    ("keyword", "Keyword"),
    ("object", "Objects"),
    ("action", "Actions"),
    ("face", "Expression"),
    # Position bonuses. Included because they feed the same total: if they were
    # omitted, the points they contribute would show up in `total - pre_boost`
    # and be reported as a multi-signal boost that never happened.
    ("beginning", "Near the start"),
    ("ending", "Near the end"),
)

# Signals that count toward the multi-signal boost. A position bonus is not
# evidence about content, and the pipeline's own boost test excludes them, so
# counting them here would over-report the signal count.
BOOST_SIGNALS = ("motion_event", "motion_peak", "audio", "keyword", "object",
                 "face")

# How many unselected peaks to report. These are the moments a user would
# adjust weights to capture, so they are the most actionable rows in the whole
# report -- and the raw material for a "this should have been included" loop.
DEFAULT_NEAR_MISS_COUNT = 5

# Resolution of the overview curves. Enough to see the shape of a feature-length
# video at page width, small enough that the arrays stay a rounding error next to
# the embedded thumbnails.
CURVE_POINTS = 480

# Resolution of one clip's own loudness strip. A clip is tens of seconds, not
# hours, so it can afford a sample every fraction of a second — and needs one,
# or the strip is a straight line that invites being read as meaning.
SEGMENT_AUDIO_POINTS = 120

# Floor for dBFS, so digital silence reports a number instead of -inf.
SILENCE_DBFS = -100.0

# Two signals firing this far apart or closer are treated as one event rather
# than a coincidence — roughly the window in which a sound and the movement
# that caused it are the same moment.
COINCIDENCE_SECONDS = 1.0


def _f(value) -> float:
    """numpy scalar or None -> plain float, so the record is JSON-serialisable."""
    if value is None:
        return 0.0
    return float(value)


def downsample(values, points: int = CURVE_POINTS) -> list:
    """Reduce a per-second array to ``points`` buckets, keeping each bucket's max.

    Max rather than mean: these curves exist to show *where the peaks are*, and
    averaging a one-second spike into a bucket of thirty erases exactly the thing
    the reader is looking for.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return []
    if arr.size <= points:
        return [round(float(v), 4) for v in arr]
    edges = np.linspace(0, arr.size, points + 1).astype(int)
    return [
        round(float(arr[lo:hi].max()), 4) if hi > lo else 0.0
        for lo, hi in zip(edges[:-1], edges[1:])
    ]


def percentile_rank(sorted_values: np.ndarray, value: float) -> float:
    """Where ``value`` sits in ``sorted_values``, as a percentage.

    Points say a moment scored; a percentile says whether it was *unusual*, and
    only the second answers "was this outstanding?".

    Ties take the midrank. Counting only what is strictly below would rank a
    video where every kept moment scored the same at the very bottom — true
    arithmetically, and a lie about the clip. Sharing the tie puts them at 50%:
    no better or worse than the rest, which is exactly the situation.
    """
    if sorted_values.size == 0:
        return 0.0
    below = float(np.searchsorted(sorted_values, value, side="left"))
    at_or_below = float(np.searchsorted(sorted_values, value, side="right"))
    return round((below + at_or_below) / 2.0 / sorted_values.size * 100.0, 1)


def to_dbfs(amplitude: float) -> float:
    """Normalised amplitude (0..1) to dBFS, floored at ``SILENCE_DBFS``.

    A real unit rather than a score: "-6 dBFS" survives a change to the weight
    table, and is the difference between "audio contributed 4 points" and
    "this is the loudest moment in the file".
    """
    if amplitude <= 0:
        return SILENCE_DBFS
    return max(SILENCE_DBFS, round(20.0 * float(np.log10(amplitude)), 1))


def split_tags(names: Iterable[str],
               composed_event_names: Optional[Iterable[str]] = None,
               ) -> tuple[list, list]:
    """Separate composed events from the raw detections they were composed from.

    The detection layer and the composition layer both end up in one per-second
    list, which makes a report row read as if a detector had fired for something
    it has no class for. ``composed_event_names`` is whatever the composition
    step produced for this run; anything else is a plain detection.
    """
    composed = set(composed_event_names or ())
    objects, events = [], []
    for name in names:
        (events if name in composed else objects).append(str(name))
    return objects, events


def boxes_by_second(bbox_cache: Optional[Iterable[Mapping]]) -> dict:
    """Index the detector's bbox cache by second.

    The cache is a list of ``{timestamp, objects, bboxes, confidences}`` records
    with boxes already normalised to ``[x, y, w, h]`` in 0..1 — which is why the
    report can draw them as a CSS overlay and never needs the frame size or an
    image library.
    """
    out: dict = {}
    for record in bbox_cache or ():
        sec = int(float(record.get("timestamp", 0)))
        names = record.get("objects") or []
        boxes = record.get("bboxes") or []
        confs = record.get("confidences") or []
        for i, box in enumerate(boxes):
            if box is None or len(box) < 4:
                continue
            out.setdefault(sec, []).append({
                "name": str(names[i]) if i < len(names) else "",
                "box": [_f(v) for v in list(box)[:4]],
                "confidence": _f(confs[i]) if i < len(confs) else 0.0,
            })
    return out


def measure_segment(*,
                    start: float,
                    end: float,
                    peak: int,
                    score: np.ndarray,
                    sorted_scores: np.ndarray,
                    signals: Mapping[str, np.ndarray],
                    sorted_signals: Mapping[str, np.ndarray],
                    amps: Sequence[float],
                    sorted_amps: np.ndarray,
                    amps_per_second: float,
                    boxes: Sequence[Mapping]) -> dict:
    """The physical facts behind one clip, not the points they earned.

    Points are a function of the weight table: change a weight and every number
    in the report moves, which makes them useless for saying what actually
    happened. Loudness in dBFS, a detector's confidence, and how far apart two
    signals fired are properties of the video, and they are what an explanation
    of *why this moment* has to be built from.
    """
    measured: dict = {
        "score_percentile": percentile_rank(sorted_scores, _f(score[peak])
                                            if peak < len(score) else 0.0),
    }

    # Loudness across the clip, in a unit that means something outside this run.
    if amps_per_second:
        lo = int(start * amps_per_second)
        hi = max(lo + 1, int(end * amps_per_second))
        window = amps[lo:hi]
        if window:
            loudest = max(window)
            measured["loudness_dbfs"] = to_dbfs(loudest)
            measured["loudness_percentile"] = percentile_rank(sorted_amps, loudest)

    # How each contributing signal ranks against the rest of the video, and the
    # second inside this clip where it fired hardest.
    per_signal = {}
    fired_at = []
    for key, _label in SIGNAL_LABELS:
        arr = signals.get(key)
        if arr is None or not len(arr):
            continue
        lo, hi = int(start), min(len(arr), int(np.ceil(end)))
        if hi <= lo:
            continue
        window = np.asarray(arr[lo:hi], dtype=float)
        if not window.any():
            continue
        best = int(np.argmax(window))
        per_signal[key] = {
            "value": _f(window[best]),
            "at": lo + best,
            "percentile": percentile_rank(sorted_signals.get(key, np.array([])),
                                          float(window[best])),
        }
        if key in BOOST_SIGNALS:
            fired_at.append(lo + best)
    if per_signal:
        measured["signals"] = per_signal

    # Signals landing together is the difference between a loud moment and a
    # loud moment you can see the cause of.
    if len(fired_at) > 1:
        spread = max(fired_at) - min(fired_at)
        measured["signal_spread_seconds"] = float(spread)
        measured["signals_coincide"] = bool(spread <= COINCIDENCE_SECONDS)

    if boxes:
        measured["detection_confidence"] = round(
            max(float(b.get("confidence") or 0.0) for b in boxes), 3)

    return measured


def peak_second(score: np.ndarray, start: float, end: float) -> int:
    """The highest-scoring second inside ``[start, end)``.

    A segment is a window built around one peak; that peak is what the
    explanation is about. Recovering it by argmax rather than threading it
    through the selection loop keeps this module independent of how segments
    were chosen.
    """
    lo = max(0, int(start))
    hi = min(len(score), max(lo + 1, int(np.ceil(end))))
    if lo >= len(score):
        return max(0, len(score) - 1)
    return lo + int(np.argmax(score[lo:hi]))


def _second_detail(sec: int,
                   score: np.ndarray,
                   signals: Mapping[str, np.ndarray],
                   object_detections: Optional[Mapping[int, Sequence[str]]],
                   actions_by_sec: Optional[Mapping[int, Sequence[tuple]]],
                   percentiles: Optional[Mapping[str, float]],
                   boost_multiplier: float,
                   min_signals_for_boost: int,
                   composed_event_names: Optional[Iterable[str]] = None) -> dict:
    """Everything known about one second, as a plain dict."""
    breakdown = {}
    for key, _label in SIGNAL_LABELS:
        arr = signals.get(key)
        breakdown[key] = _f(arr[sec]) if arr is not None and sec < len(arr) else 0.0

    pre_boost = sum(breakdown.values())
    total = _f(score[sec]) if sec < len(score) else 0.0

    detected = list(object_detections.get(sec, [])) if object_detections else []
    objects, events = split_tags(detected, composed_event_names)

    actions = []
    raw_actions = actions_by_sec.get(sec, []) if actions_by_sec else []
    for name, conf in raw_actions:
        tier = None
        if percentiles:
            p90 = percentiles.get("90th")
            p50 = percentiles.get("50th")
            if p90 is not None and conf >= p90:
                tier = "bonus"
            elif p50 is not None and conf >= p50:
                tier = "normal"
            else:
                tier = "reduced"
        actions.append({"name": str(name), "confidence": _f(conf), "tier": tier})

    # Mirrors the pipeline's own count: a signal contributes if it scored, and
    # actions count by presence rather than points (an action can be detected
    # but score zero when "require objects" suppresses it).
    contributing = [
        key for key in BOOST_SIGNALS if breakdown.get(key, 0.0) > 0
    ]
    if raw_actions:
        contributing.append("action")

    boost_points = total - pre_boost
    boosted = len(contributing) >= min_signals_for_boost and boost_points > 1e-9

    return {
        "second": int(sec),
        "timestamp": format_timestamp(sec),
        "score": total,
        "pre_boost_score": pre_boost,
        "breakdown": breakdown,
        "objects": objects,
        "events": events,
        "actions": actions,
        "signals_present": contributing,
        "boost": {
            "applied": bool(boosted),
            "signal_count": len(contributing),
            "multiplier": _f(boost_multiplier) if boosted else 1.0,
            "points": _f(boost_points) if boosted else 0.0,
        },
    }


def format_timestamp(seconds: float) -> str:
    """Seconds -> ``M:SS`` or ``H:MM:SS`` once past an hour."""
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def build_report(*,
                 video_path: str,
                 video_duration: float,
                 score: np.ndarray,
                 signals: Mapping[str, np.ndarray],
                 segments: Sequence[tuple],
                 object_detections: Optional[Mapping[int, Sequence[str]]] = None,
                 actions_by_sec: Optional[Mapping[int, Sequence[tuple]]] = None,
                 action_percentiles: Optional[Mapping[str, Mapping[str, float]]] = None,
                 settings: Optional[Mapping] = None,
                 boost_multiplier: float = 1.0,
                 min_signals_for_boost: int = 2,
                 near_miss_count: int = DEFAULT_NEAR_MISS_COUNT,
                 thumbnail_fn: Optional[Callable[[float], Optional[bytes]]] = None,
                 composed_event_names: Optional[Iterable[str]] = None,
                 waveform: Optional[Sequence] = None,
                 bbox_cache: Optional[Iterable[Mapping]] = None,
                 ) -> dict:
    """Attribute every kept segment to the evidence that selected it.

    ``signals`` maps the keys in :data:`SIGNAL_LABELS` to per-second arrays;
    absent keys are simply reported as zero, so a caller that never ran a
    detector does not have to fabricate one.

    ``action_percentiles`` is keyed by action name (the pipeline computes them
    per action type) and is used only to label a confidence tier.

    ``composed_event_names`` lets the report tell a composed event apart from the
    raw detections it was built out of; see :func:`split_tags`.

    ``waveform`` is the loudness envelope of the whole video — the caller's
    existing ``extract_waveform_data`` output, either ``(min, max, rms)`` triples
    or bare amplitudes. It is reported even when audio contributed no points,
    because "was anything happening acoustically here?" is context the reader
    wants whether or not the scoring used it.

    Near-misses are the highest-scoring seconds *not* covered by any kept
    segment. They are what a user would tune the weights to capture, so they are
    reported alongside the selections rather than hidden.
    """
    score = np.asarray(score, dtype=float)
    segments = [(float(s), float(e)) for s, e in segments]
    segments.sort(key=lambda x: x[0])
    boxes_at = boxes_by_second(bbox_cache)

    # Flatten the loudness envelope once, at full resolution. Each clip then
    # takes its own slice of *this* rather than of the page-wide curve: at 480
    # points a feature-length video gives a 30-second clip about four samples,
    # which draws as a straight line and says nothing.
    amps: list = []
    if waveform is not None:
        try:
            first = next(iter(waveform))
        except StopIteration:
            first = None
        if isinstance(first, (tuple, list, np.ndarray)):
            # (min, max, rms) triples: rms is the perceptual one.
            amps = [abs(float(p[-1])) for p in waveform]
        elif first is not None:
            amps = [abs(float(p)) for p in waveform]
    amps_per_second = (len(amps) / video_duration) if (amps and video_duration) else 0.0

    # Sorted once for the percentile lookups: every clip is ranked against the
    # same distributions, and sorting per clip would be the expensive part.
    # Zeros are excluded — a moment is unusual relative to the video's activity,
    # not relative to the silence between it.
    sorted_scores = np.sort(score[score > 0])
    sorted_amps = np.sort(np.asarray([a for a in amps if a > 0], dtype=float))
    sorted_signals = {}
    for key, _label in SIGNAL_LABELS:
        arr = signals.get(key)
        if arr is not None and len(arr):
            arr = np.asarray(arr, dtype=float)
            sorted_signals[key] = np.sort(arr[arr > 0])

    def detail_for(sec: int) -> dict:
        pcts = None
        if action_percentiles and actions_by_sec and sec in actions_by_sec:
            names = [n for n, _ in actions_by_sec[sec]]
            if names:
                pcts = action_percentiles.get(names[0])
        return _second_detail(sec, score, signals, object_detections,
                              actions_by_sec, pcts, boost_multiplier,
                              min_signals_for_boost, composed_event_names)

    covered: set = set()
    entries = []
    for i, (start, end) in enumerate(segments, start=1):
        sec = peak_second(score, start, end)
        covered.update(range(int(start), int(np.ceil(end))))
        entry = detail_for(sec)
        if boxes_at.get(sec):
            entry["boxes"] = boxes_at[sec]
        entry["measured"] = measure_segment(
            start=start, end=end, peak=sec,
            score=score, sorted_scores=sorted_scores,
            signals=signals, sorted_signals=sorted_signals,
            amps=amps, sorted_amps=sorted_amps,
            amps_per_second=amps_per_second,
            boxes=boxes_at.get(sec, ()),
        )
        if amps_per_second:
            lo = int(start * amps_per_second)
            hi = max(lo + 1, int(end * amps_per_second))
            entry["audio"] = downsample(amps[lo:hi], points=SEGMENT_AUDIO_POINTS)
        entry.update({
            "index": i,
            "start": start,
            "end": end,
            "duration": end - start,
            "range": f"{format_timestamp(start)} – {format_timestamp(end)}",
        })
        if thumbnail_fn is not None:
            try:
                raw = thumbnail_fn(sec)
            except Exception:
                raw = None
            if raw:
                entry["thumbnail"] = ("data:image/jpeg;base64,"
                                      + base64.b64encode(raw).decode("ascii"))
        entries.append(entry)

    near_misses = []
    if near_miss_count > 0 and len(score):
        for sec in np.argsort(score)[::-1]:
            sec = int(sec)
            if sec in covered or score[sec] <= 0:
                continue
            if any(abs(sec - n["second"]) < 5 for n in near_misses):
                continue          # one row per cluster, not five adjacent seconds
            near_misses.append(detail_for(sec))
            if len(near_misses) >= near_miss_count:
                break

    # What the whole cut was actually built out of. A report where one signal
    # supplies every point is a report about a misconfigured weight table, and
    # that is only visible in aggregate.
    signal_totals = {}
    for key, _label in SIGNAL_LABELS:
        total = sum(e["breakdown"].get(key, 0.0) for e in entries)
        if total:
            signal_totals[key] = _f(total)

    audio_curve = downsample(amps) if amps else []

    kept_duration = sum(e - s for s, e in segments)
    return {
        "schema": 2,
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "video": {
            "path": str(video_path),
            "name": str(video_path).replace("\\", "/").rsplit("/", 1)[-1],
            "duration": _f(video_duration),
        },
        "totals": {
            "segments": len(segments),
            "duration": _f(kept_duration),
            "coverage_pct": _f(100.0 * kept_duration / video_duration) if video_duration else 0.0,
        },
        "settings": dict(settings or {}),
        "signal_totals": signal_totals,
        "curves": {
            "points": CURVE_POINTS,
            "score": downsample(score),
            # Full resolution, not just the drawing. Selection runs off this
            # array, so keeping it is what lets a later "give me a different
            # clip" re-choose without the video, the detectors or a re-run —
            # about 25 KB an hour, against ~400 KB of thumbnails.
            "score_per_second": [round(float(v), 3) for v in score],
            "audio": audio_curve,
            # Per-clip strips are drawn against this, not against their own max,
            # so a quiet clip reads as quiet instead of being stretched to full
            # height and looking exactly like the loudest moment in the video.
            "audio_peak": round(float(max(amps)), 4) if amps else 0.0,
        },
        "segments": entries,
        "near_misses": near_misses,
    }


def score_from_report(report: Mapping) -> np.ndarray:
    """The per-second score the cut was made from.

    With this and :func:`segments_from_report`, a saved report is enough to
    re-choose a clip — the selection is arithmetic over this array, so nothing
    has to be re-detected and the video does not have to be present.
    """
    return np.asarray(
        (report.get("curves") or {}).get("score_per_second") or [], dtype=float
    )


def segments_from_report(report: Mapping) -> list[tuple[float, float]]:
    """The kept ranges, in the order the record lists them."""
    return [(float(e["start"]), float(e["end"])) for e in report.get("segments", [])]


# --------------------------------------------------------------------------- #
# Renderers
# --------------------------------------------------------------------------- #

def render_text(report: Mapping) -> str:
    """The debug-log view — the same breakdown the pipeline used to print."""
    out = ["=== HIGHLIGHT BREAKDOWN ==="]
    t = report["totals"]
    out.append(f"{t['segments']} segment(s), {t['duration']:.1f}s "
               f"({t['coverage_pct']:.1f}% of the source)")

    for e in report["segments"]:
        out.append("")
        out.append(f"[{e['index']}] {e['range']}  peak {e['timestamp']} "
                   f"({e['second']}s): {e['score']:.1f} points")
        for key, label in SIGNAL_LABELS:
            value = e["breakdown"].get(key, 0.0)
            if value:
                out.append(f"    {label}: {value:.1f}")
        if e["objects"]:
            out.append(f"    Objects detected: {', '.join(e['objects'])}")
        if e.get("events"):
            out.append(f"    Events composed: {', '.join(e['events'])}")
        for a in e["actions"]:
            tier = f" [{a['tier']}]" if a["tier"] else ""
            out.append(f"    Action: {a['name']} ({a['confidence']:.2f}){tier}")
        b = e["boost"]
        if b["applied"]:
            out.append(f"    Multi-signal boost: {b['signal_count']} signals "
                       f"x{b['multiplier']} -> +{b['points']:.1f}")

    if report["near_misses"]:
        out.append("")
        out.append("--- Highest-scoring moments NOT included ---")
        for e in report["near_misses"]:
            out.append(f"    {e['timestamp']} ({e['second']}s): {e['score']:.1f} points"
                       + (f"  [{', '.join(e['signals_present'])}]"
                          if e["signals_present"] else ""))
    return "\n".join(out)


_CSS = """
:root{--bg:#141416;--card:#1c1c20;--line:#2a2a30;--text:#e8e8ea;--dim:#9a9aa2;
      --accent:#5ac8b0;--warm:#e8a33d;--cool:#7aa7e8}
*{box-sizing:border-box}
body{margin:0;padding:32px 20px;background:var(--bg);color:var(--text);
     font:15px/1.55 -apple-system,Segoe UI,Roboto,sans-serif}
.wrap{max-width:900px;margin:0 auto}
h1{font-size:22px;margin:0 0 4px}
.sub{color:var(--dim);font-size:13.5px;margin-bottom:8px}
.sub2{color:var(--text);font-size:15px;margin-bottom:24px}
.totals{display:flex;gap:28px;flex-wrap:wrap;padding:16px 0 24px;
        border-bottom:1px solid var(--line);margin-bottom:28px}
.totals div span{display:block}
.totals .n{font-size:24px;color:var(--accent)}
.totals .l{font-size:12px;color:var(--dim)}
.seg{background:var(--card);border:1px solid var(--line);border-radius:10px;
     padding:16px;margin-bottom:14px;display:flex;gap:16px}
.shotwrap{width:200px;flex-shrink:0;align-self:flex-start}
/* The box overlay is positioned against this element, so it must wrap the
   image and nothing else — a caption inside it would skew every percentage. */
.shot{position:relative}
.seg img{width:100%;display:block;border-radius:6px}
.bx{position:absolute;border:1.5px solid var(--cool);border-radius:2px;
    pointer-events:none}
.bx.evt{border-color:var(--warm)}
.bx b{position:absolute;top:-13px;left:-1.5px;font:9.5px/1.35 ui-monospace,monospace;
      font-weight:600;background:var(--cool);color:#101014;padding:0 3px;
      border-radius:2px;white-space:nowrap}
.bx.evt b{background:var(--warm)}
.shotlab{color:var(--dim);font-size:11px;margin-top:4px;text-align:center}
.seg .body{flex:1;min-width:0}
.rng{font-weight:600}
.pts{color:var(--accent);font-weight:600}
.meta{color:var(--dim);font-size:13px;margin:2px 0 10px}
.says{color:var(--text);font-size:13.5px;margin:10px 0 2px}
.meas{color:var(--cool);font-size:12px;margin:2px 0}
.bar{display:flex;align-items:center;gap:10px;margin:3px 0;font-size:13px}
.bar .lab{width:110px;color:var(--dim);flex-shrink:0}
.bar .track{flex:1;height:8px;background:#26262c;border-radius:4px;overflow:hidden}
.bar .fill{height:100%;background:var(--accent)}
.bar .val{width:42px;text-align:right;color:var(--dim)}
.tags{margin-top:10px;display:flex;flex-wrap:wrap;gap:6px;align-items:center}
.tags .kind{font-size:11px;color:var(--dim);text-transform:uppercase;
            letter-spacing:.06em;width:56px;flex-shrink:0}
.tag{font-size:12px;padding:2px 8px;border-radius:999px;
     border:1px solid var(--line);color:var(--dim)}
.tag.act{border-color:var(--accent);color:var(--accent)}
.tag.evt{border-color:var(--warm);color:var(--warm)}
.tag.obj{border-color:var(--cool);color:var(--cool)}
/* Overview */
.tl{margin:0 0 28px}
.tl svg{display:block;width:100%;height:96px}
.tl .cap{display:flex;justify-content:space-between;color:var(--dim);
         font-size:11.5px;margin-top:4px}
.legend{display:flex;gap:16px;flex-wrap:wrap;color:var(--dim);font-size:12px;
        margin-top:8px}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;
          margin-right:5px;vertical-align:-1px}
/* Advisor findings */
.find{background:var(--card);border:1px solid var(--line);border-left-width:3px;
      border-radius:8px;padding:12px 14px;margin-bottom:10px}
.find.sev-high{border-left-color:#e8685d}
.find.sev-medium{border-left-color:var(--warm)}
.find.sev-low{border-left-color:var(--cool)}
.find .fh{font-weight:600;margin-bottom:6px}
.find .sev{font-size:10.5px;text-transform:uppercase;letter-spacing:.06em;
           color:var(--dim);margin-right:8px}
.find p{margin:4px 0;font-size:13.5px;color:var(--dim)}
.find .fix{color:var(--text)}
.narr{background:#191a1f;border:1px solid var(--line);border-radius:8px;
      padding:12px 14px;margin-bottom:12px;white-space:pre-wrap;font-size:13.5px}
.wave{display:block;width:100%;height:34px;margin-top:8px;opacity:.85}
.wavelab{color:var(--dim);font-size:11.5px;margin-top:2px}
.boost{margin-top:10px;font-size:12.5px;color:var(--warm)}
h2{font-size:16px;margin:34px 0 6px}
.note{color:var(--dim);font-size:13px;margin-bottom:14px}
table{width:100%;border-collapse:collapse;font-size:13.5px}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line)}
th{color:var(--dim);font-weight:500;font-size:12px}
.scroll{overflow-x:auto}
@media(max-width:640px){.seg{flex-direction:column}.shotwrap{width:100%}}
"""


def _bars(entry: Mapping, max_points: float) -> str:
    rows = []
    for key, label in SIGNAL_LABELS:
        value = entry["breakdown"].get(key, 0.0)
        if not value:
            continue
        pct = (value / max_points * 100.0) if max_points else 0.0
        rows.append(
            f'<div class="bar"><span class="lab">{html.escape(label)}</span>'
            f'<span class="track"><span class="fill" style="width:{pct:.1f}%"></span></span>'
            f'<span class="val">{value:.0f}</span></div>'
        )
    return "".join(rows)


def _area_path(values: Sequence[float], width: float, height: float,
               baseline: float, peak: Optional[float] = None) -> str:
    """A closed SVG path for a filled area chart.

    ``peak`` is the value drawn at full height; without one the series is scaled
    to its own maximum. Passing a shared peak is what makes several strips
    comparable with each other.

    Fills only, no strokes: the SVG is stretched to page width with
    ``preserveAspectRatio="none"``, which would smear a stroke into a wedge.
    """
    if not values:
        return ""
    peak = peak or max(values) or 1.0
    step = width / max(1, len(values) - 1)
    points = [
        f"{i * step:.2f},{baseline - (v / peak) * height:.2f}"
        for i, v in enumerate(values)
    ]
    return f"M0,{baseline:.2f} L" + " L".join(points) + f" L{width:.2f},{baseline:.2f} Z"


def _run_sentence(report: Mapping) -> str:
    """The one-line reading of the whole run, above the totals."""
    try:
        from modules.highlight_prose import summarise_run
        return summarise_run(report)
    except Exception:
        return ""


def _overview(report: Mapping) -> str:
    """Where the kept moments sit in the video, against the score curve.

    The single most-asked question a highlight raises is "did it look at the
    whole video, or just one stretch of it?", and no amount of per-segment detail
    answers it. One strip does.
    """
    duration = report["video"]["duration"] or 1.0
    curves = report.get("curves") or {}
    score_curve = curves.get("score") or []
    audio_curve = curves.get("audio") or []

    W, H = 1000.0, 96.0
    curve_h, band_y, band_h = 56.0, 66.0, 16.0

    parts = [
        f'<svg viewBox="0 0 {W:.0f} {H:.0f}" preserveAspectRatio="none" '
        f'role="img" aria-label="Where the highlights fall in the video">'
    ]
    if score_curve:
        parts.append(
            f'<path d="{_area_path(score_curve, W, curve_h, curve_h + 4)}" '
            f'fill="#2f3a44"/>'
        )
    parts.append(f'<rect x="0" y="{band_y}" width="{W:.0f}" height="{band_h}" '
                 f'fill="#26262c" rx="3"/>')

    for e in report["segments"]:
        x = max(0.0, e["start"] / duration * W)
        w = max(2.0, (e["end"] - e["start"]) / duration * W)
        parts.append(f'<rect x="{x:.2f}" y="{band_y}" width="{w:.2f}" '
                     f'height="{band_h}" fill="#5ac8b0"/>')
    for e in report.get("near_misses", []):
        x = max(0.0, e["second"] / duration * W)
        parts.append(f'<rect x="{x:.2f}" y="{band_y}" width="2.5" '
                     f'height="{band_h}" fill="#e8a33d"/>')
    parts.append("</svg>")

    mid = format_timestamp(duration / 2)
    caption = (f'<div class="cap"><span>0:00</span><span>{html.escape(mid)}</span>'
               f'<span>{html.escape(format_timestamp(duration))}</span></div>')

    wave = ""
    if audio_curve:
        wave = (
            f'<svg class="wave" viewBox="0 0 {W:.0f} 34" preserveAspectRatio="none" '
            f'role="img" aria-label="Loudness across the video">'
            f'<path d="{_area_path(audio_curve, W, 30.0, 32.0)}" fill="#3a3a46"/>'
            f'</svg>'
            '<div class="wavelab">Loudness across the whole video, on the same '
            'time axis as the strip above — shown for context whether or not '
            'audio contributed points.</div>'
        )

    legend = ('<div class="legend">'
              '<span><i style="background:#2f3a44"></i>score</span>'
              '<span><i style="background:#5ac8b0"></i>kept</span>'
              '<span><i style="background:#e8a33d"></i>scored well, not kept</span>'
              '</div>')

    return f'<div class="tl">{"".join(parts)}{caption}{wave}{legend}</div>'


def _summary(report: Mapping) -> str:
    """What the whole cut was built out of, by signal."""
    totals = report.get("signal_totals") or {}
    if not totals:
        return ""
    labels = dict(SIGNAL_LABELS)
    biggest = max(totals.values()) or 1.0
    rows = "".join(
        f'<div class="bar"><span class="lab">'
        f'{html.escape(labels.get(key, key))}</span>'
        f'<span class="track"><span class="fill" '
        f'style="width:{value / biggest * 100.0:.1f}%"></span></span>'
        f'<span class="val">{value:.0f}</span></div>'
        for key, value in sorted(totals.items(), key=lambda kv: -kv[1])
    )
    note = ""
    if len(totals) == 1:
        only = labels.get(next(iter(totals)), next(iter(totals)))
        note = (f'<p class="note">Every point in this highlight came from '
                f'<b>{html.escape(only)}</b>. The other signals are switched off '
                f'or weighted at zero — nothing else could influence the cut.</p>')
    return f'<h2>What decided the cut</h2>{note}{rows}'


def _advice(report: Mapping) -> str:
    """What to change, if anything diagnosed this run (see modules.advisor)."""
    findings = report.get("advice") or []
    if not findings:
        return ""
    rows = []
    for f in findings:
        severity = str(f.get("severity", "low"))
        rows.append(
            f'<div class="find sev-{html.escape(severity)}">'
            f'<div class="fh"><span class="sev">{html.escape(severity)}</span>'
            f'{html.escape(str(f.get("title", "")))}</div>'
            f'<p>{html.escape(str(f.get("detail", "")))}</p>'
            f'<p class="fix"><b>Try:</b> {html.escape(str(f.get("remedy", "")))}</p>'
            f'</div>'
        )
    narration = ""
    if report.get("advice_narration"):
        narration = (f'<div class="narr">'
                     f'{html.escape(str(report["advice_narration"]))}</div>')
    return (
        '<h2>What to try next</h2>'
        '<p class="note">Worked out from this run\'s own numbers — each point '
        'below is backed by the figures shown with it, not by a guess about '
        'what you meant.</p>'
        f'{narration}{"".join(rows)}'
    )


def _measurements(entry: Mapping, peer_scores=None) -> str:
    """What was physically true here, in units that outlive the weight table.

    Led by the plain-language reading of those same numbers: a row of figures
    is precise and mostly unread, and the sentence above them is what a person
    actually takes away.
    """
    m = entry.get("measured") or {}
    if not m:
        return ""

    from modules.highlight_prose import describe

    sentence = describe(entry, peer_scores)
    lead = (f'<div class="says">{html.escape(sentence)}</div>' if sentence else "")

    parts = []
    pct = m.get("score_percentile")
    if pct is not None:
        parts.append(f"scored above {pct:.0f}% of the video")
    if "loudness_dbfs" in m:
        loud = f"peaked at {m['loudness_dbfs']:.0f} dBFS"
        if "loudness_percentile" in m:
            loud += f" ({m['loudness_percentile']:.0f}th pct)"
        parts.append(loud)
    if m.get("signals_coincide"):
        spread = m.get("signal_spread_seconds", 0.0)
        parts.append("signals landed together"
                     if spread <= 0 else f"signals within {spread:.0f}s")
    elif "signal_spread_seconds" in m:
        parts.append(f"signals {m['signal_spread_seconds']:.0f}s apart")
    if "detection_confidence" in m:
        parts.append(f"best detection {m['detection_confidence']:.2f}")

    if not parts:
        return lead
    return lead + f'<div class="meas">{html.escape(" · ".join(parts))}</div>'


def _tag_rows(entry: Mapping) -> str:
    """Tags grouped by what produced them, one row per kind."""
    groups = (
        ("objects", "obj", [html.escape(str(o)) for o in entry.get("objects", [])]),
        ("events", "evt", [html.escape(str(v)) for v in entry.get("events", [])]),
        ("actions", "act", [
            f'{html.escape(str(a["name"]))} {a["confidence"]:.2f}'
            for a in entry.get("actions", [])
        ]),
    )
    rows = []
    for kind, css, items in groups:
        if not items:
            continue
        tags = "".join(f'<span class="tag {css}">{item}</span>' for item in items)
        rows.append(f'<div class="tags"><span class="kind">{kind}</span>{tags}</div>')
    return "".join(rows)


def _shot(entry: Mapping, composed: frozenset) -> str:
    """The thumbnail, with what the detector saw drawn over it.

    Boxes are normalised, so percentage positioning reproduces them at any
    thumbnail width without the report ever touching an image library.
    """
    if not entry.get("thumbnail"):
        return ""
    overlay = []
    for box in entry.get("boxes", []):
        x, y, w, h = box["box"]
        if w <= 0 or h <= 0:
            continue
        kind = " evt" if box["name"] in composed else ""
        label = html.escape(box["name"]) if box["name"] else ""
        overlay.append(
            f'<span class="bx{kind}" style="left:{x * 100:.1f}%;top:{y * 100:.1f}%;'
            f'width:{w * 100:.1f}%;height:{h * 100:.1f}%">'
            f'{f"<b>{label}</b>" if label else ""}</span>'
        )
    caption = ('<div class="shotlab">boxes as detected at the peak second</div>'
               if overlay else "")
    return (f'<div class="shotwrap"><div class="shot">'
            f'<img src="{entry["thumbnail"]}" alt="">'
            f'{"".join(overlay)}</div>{caption}</div>')


def _segment_wave(report: Mapping, entry: Mapping) -> str:
    """The loudness envelope under one clip, with its peak second marked."""
    curves = report.get("curves") or {}
    audio = curves.get("audio") or []
    duration = report["video"]["duration"] or 0.0
    # The clip's own full-resolution slice when the record has one; the coarse
    # page-wide curve only as a fallback for an older report.
    window = entry.get("audio") or []
    if not window:
        if not audio or duration <= 0:
            return ""
        lo = int(entry["start"] / duration * len(audio))
        hi = max(lo + 2, int(entry["end"] / duration * len(audio)))
        window = audio[lo:hi]
    if not window:
        return ""

    W = 400.0
    marker = ""
    span = entry["end"] - entry["start"]
    if span > 0:
        pos = min(1.0, max(0.0, (entry["second"] - entry["start"]) / span))
        marker = (f'<rect x="{pos * W:.1f}" y="0" width="1.5" height="26" '
                  f'fill="#5ac8b0" opacity=".8"/>')

    scored = entry["breakdown"].get("audio", 0.0)
    note = (f"contributed {scored:.0f} points" if scored
            else "contributed no points to this pick")
    caption = (
        f'<div class="wavelab">Loudness through the clip — '
        f'<b style="color:#5ac8b0">|</b> marks {html.escape(entry["timestamp"])}, '
        f'the second that scored highest. Volume is drawn for context only and '
        f'{note}, so a louder stretch elsewhere in the clip did not move it.</div>'
    )
    return (
        f'<svg class="wave" viewBox="0 0 {W:.0f} 26" preserveAspectRatio="none" '
        f'role="img" aria-label="Loudness during this clip">'
        f'<path d="{_area_path(window, W, 22.0, 24.0, peak=(curves.get("audio_peak") or None))}" '
        f'fill="#3a3a46"/>'
        f'{marker}</svg>{caption}'
    )


def render_html(report: Mapping, title: Optional[str] = None) -> str:
    """A standalone page: inline CSS, embedded thumbnails, nothing fetched."""
    video = report["video"]
    totals = report["totals"]
    heading = title or f"Why these moments — {video['name']}"

    max_points = max([e["score"] for e in report["segments"]] or [1.0])
    # Each clip is described relative to the others in the cut.
    peer_scores = [float(e.get("score") or 0.0) for e in report["segments"]]
    composed = frozenset(
        name for e in report["segments"] for name in e.get("events", [])
    )

    segs = []
    for e in report["segments"]:
        thumb = _shot(e, composed)
        tags = _tag_rows(e)
        boost = ""
        if e["boost"]["applied"]:
            boost = (f'<div class="boost">Multi-signal boost — '
                     f'{e["boost"]["signal_count"]} signals agreed, '
                     f'×{e["boost"]["multiplier"]:g} (+{e["boost"]["points"]:.0f})</div>')
        segs.append(
            f'<div class="seg">{thumb}<div class="body">'
            f'<div><span class="rng">{html.escape(e["range"])}</span> · '
            f'<span class="pts">{e["score"]:.0f} points</span></div>'
            f'<div class="meta">peak at {html.escape(e["timestamp"])} · '
            f'{e["duration"]:.0f}s long</div>'
            f'{_bars(e, max_points)}'
            f'{_measurements(e, peer_scores)}'
            f'{tags}'
            f'{_segment_wave(report, e)}'
            f'{boost}</div></div>'
        )

    near = ""
    if report["near_misses"]:
        rows = "".join(
            f'<tr><td>{html.escape(e["timestamp"])}</td>'
            f'<td>{e["score"]:.0f}</td>'
            f'<td>{html.escape(", ".join(e["signals_present"]) or "—")}</td>'
            f'<td>{html.escape(", ".join(e["objects"]) or "—")}</td>'
            f'<td>{html.escape(", ".join(e.get("events", [])) or "—")}</td></tr>'
            for e in report["near_misses"]
        )
        near = (
            '<h2>Scored well, but not included</h2>'
            '<p class="note">The highest-scoring moments that did not make the cut — '
            'usually because the highlight was already full, or a neighbouring '
            'second scored higher. Raise the weight of a signal below to pull '
            'moments like these in.</p>'
            '<div class="scroll"><table><thead><tr><th>Time</th><th>Points</th>'
            '<th>Signals</th><th>Objects</th><th>Events</th></tr></thead>'
            f'<tbody>{rows}</tbody></table></div>'
        )

    settings = ""
    if report.get("settings"):
        rows = "".join(
            f'<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>'
            for k, v in sorted(report["settings"].items())
        )
        settings = ('<h2>Settings used</h2><div class="scroll"><table>'
                    f'<tbody>{rows}</tbody></table></div>')

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(heading)}</title><style>{_CSS}</style></head>
<body><div class="wrap">
<h1>{html.escape(heading)}</h1>
<div class="sub">Generated {html.escape(report["generated_at"])}</div>
<div class="sub2">{html.escape(_run_sentence(report))}</div>
<div class="totals">
  <div><span class="n">{totals["segments"]}</span><span class="l">segments kept</span></div>
  <div><span class="n">{totals["duration"]:.0f}s</span><span class="l">total length</span></div>
  <div><span class="n">{totals["coverage_pct"]:.1f}%</span><span class="l">of the source</span></div>
</div>
{_overview(report)}
{_advice(report)}
{_summary(report)}
<h2>The moments, in order</h2>
<p class="note">One row per clip that was kept. The bars are the points that
second earned, broken down by signal; the tags are what was detected there,
grouped by what produced them — <b>objects</b> come straight from the detector,
<b>events</b> are combinations the composition rules recognised, <b>actions</b>
come from the action model with their confidence.</p>
{"".join(segs)}
{near}
{settings}
</div></body></html>
"""


def write_report(report: Mapping, html_path: str,
                 json_path: Optional[str] = None,
                 title: Optional[str] = None) -> None:
    """Write the page, and the record it was rendered from.

    The JSON is not a debugging leftover: it is the structured form a later
    "this pick was wrong" signal has to attach to, and rebuilding it from HTML
    would be absurd.
    """
    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(render_html(report, title=title))
    if json_path:
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1)
