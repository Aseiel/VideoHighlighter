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
    # Position bonuses. Included because they feed the same total: if they were
    # omitted, the points they contribute would show up in `total - pre_boost`
    # and be reported as a multi-signal boost that never happened.
    ("beginning", "Near the start"),
    ("ending", "Near the end"),
)

# Signals that count toward the multi-signal boost. A position bonus is not
# evidence about content, and the pipeline's own boost test excludes them, so
# counting them here would over-report the signal count.
BOOST_SIGNALS = ("motion_event", "motion_peak", "audio", "keyword", "object")

# How many unselected peaks to report. These are the moments a user would
# adjust weights to capture, so they are the most actionable rows in the whole
# report -- and the raw material for a "this should have been included" loop.
DEFAULT_NEAR_MISS_COUNT = 5


def _f(value) -> float:
    """numpy scalar or None -> plain float, so the record is JSON-serialisable."""
    if value is None:
        return 0.0
    return float(value)


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
                   min_signals_for_boost: int) -> dict:
    """Everything known about one second, as a plain dict."""
    breakdown = {}
    for key, _label in SIGNAL_LABELS:
        arr = signals.get(key)
        breakdown[key] = _f(arr[sec]) if arr is not None and sec < len(arr) else 0.0

    pre_boost = sum(breakdown.values())
    total = _f(score[sec]) if sec < len(score) else 0.0

    objects = list(object_detections.get(sec, [])) if object_detections else []

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
                 ) -> dict:
    """Attribute every kept segment to the evidence that selected it.

    ``signals`` maps the keys in :data:`SIGNAL_LABELS` to per-second arrays;
    absent keys are simply reported as zero, so a caller that never ran a
    detector does not have to fabricate one.

    ``action_percentiles`` is keyed by action name (the pipeline computes them
    per action type) and is used only to label a confidence tier.

    Near-misses are the highest-scoring seconds *not* covered by any kept
    segment. They are what a user would tune the weights to capture, so they are
    reported alongside the selections rather than hidden.
    """
    score = np.asarray(score, dtype=float)
    segments = [(float(s), float(e)) for s, e in segments]
    segments.sort(key=lambda x: x[0])

    def detail_for(sec: int) -> dict:
        pcts = None
        if action_percentiles and actions_by_sec and sec in actions_by_sec:
            names = [n for n, _ in actions_by_sec[sec]]
            if names:
                pcts = action_percentiles.get(names[0])
        return _second_detail(sec, score, signals, object_detections,
                              actions_by_sec, pcts, boost_multiplier,
                              min_signals_for_boost)

    covered: set = set()
    entries = []
    for i, (start, end) in enumerate(segments, start=1):
        sec = peak_second(score, start, end)
        covered.update(range(int(start), int(np.ceil(end))))
        entry = detail_for(sec)
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

    kept_duration = sum(e - s for s, e in segments)
    return {
        "schema": 1,
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
        "segments": entries,
        "near_misses": near_misses,
    }


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
      --accent:#5ac8b0;--warm:#e8a33d}
*{box-sizing:border-box}
body{margin:0;padding:32px 20px;background:var(--bg);color:var(--text);
     font:15px/1.55 -apple-system,Segoe UI,Roboto,sans-serif}
.wrap{max-width:900px;margin:0 auto}
h1{font-size:22px;margin:0 0 4px}
.sub{color:var(--dim);font-size:13.5px;margin-bottom:28px}
.totals{display:flex;gap:28px;flex-wrap:wrap;padding:16px 0 24px;
        border-bottom:1px solid var(--line);margin-bottom:28px}
.totals div span{display:block}
.totals .n{font-size:24px;color:var(--accent)}
.totals .l{font-size:12px;color:var(--dim)}
.seg{background:var(--card);border:1px solid var(--line);border-radius:10px;
     padding:16px;margin-bottom:14px;display:flex;gap:16px}
.seg img{width:160px;height:90px;object-fit:cover;border-radius:6px;flex-shrink:0}
.seg .body{flex:1;min-width:0}
.rng{font-weight:600}
.pts{color:var(--accent);font-weight:600}
.meta{color:var(--dim);font-size:13px;margin:2px 0 10px}
.bar{display:flex;align-items:center;gap:10px;margin:3px 0;font-size:13px}
.bar .lab{width:110px;color:var(--dim);flex-shrink:0}
.bar .track{flex:1;height:8px;background:#26262c;border-radius:4px;overflow:hidden}
.bar .fill{height:100%;background:var(--accent)}
.bar .val{width:42px;text-align:right;color:var(--dim)}
.tags{margin-top:10px;display:flex;flex-wrap:wrap;gap:6px}
.tag{font-size:12px;padding:2px 8px;border-radius:999px;
     border:1px solid var(--line);color:var(--dim)}
.tag.act{border-color:var(--accent);color:var(--accent)}
.boost{margin-top:10px;font-size:12.5px;color:var(--warm)}
h2{font-size:16px;margin:34px 0 6px}
.note{color:var(--dim);font-size:13px;margin-bottom:14px}
table{width:100%;border-collapse:collapse;font-size:13.5px}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line)}
th{color:var(--dim);font-weight:500;font-size:12px}
.scroll{overflow-x:auto}
@media(max-width:640px){.seg{flex-direction:column}.seg img{width:100%;height:auto}}
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


def render_html(report: Mapping, title: Optional[str] = None) -> str:
    """A standalone page: inline CSS, embedded thumbnails, nothing fetched."""
    video = report["video"]
    totals = report["totals"]
    heading = title or f"Why these moments — {video['name']}"

    max_points = max([e["score"] for e in report["segments"]] or [1.0])

    segs = []
    for e in report["segments"]:
        thumb = (f'<img src="{e["thumbnail"]}" alt="">' if e.get("thumbnail") else "")
        tags = "".join(
            f'<span class="tag">{html.escape(o)}</span>' for o in e["objects"]
        ) + "".join(
            f'<span class="tag act">{html.escape(a["name"])} '
            f'{a["confidence"]:.2f}</span>' for a in e["actions"]
        )
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
            f'{f"<div class=tags>{tags}</div>" if tags else ""}'
            f'{boost}</div></div>'
        )

    near = ""
    if report["near_misses"]:
        rows = "".join(
            f'<tr><td>{html.escape(e["timestamp"])}</td>'
            f'<td>{e["score"]:.0f}</td>'
            f'<td>{html.escape(", ".join(e["signals_present"]) or "—")}</td>'
            f'<td>{html.escape(", ".join(e["objects"]) or "—")}</td></tr>'
            for e in report["near_misses"]
        )
        near = (
            '<h2>Scored well, but not included</h2>'
            '<p class="note">The highest-scoring moments that did not make the cut — '
            'usually because the highlight was already full, or a neighbouring '
            'second scored higher. Raise the weight of a signal below to pull '
            'moments like these in.</p>'
            '<div class="scroll"><table><thead><tr><th>Time</th><th>Points</th>'
            '<th>Signals</th><th>Objects</th></tr></thead>'
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
<div class="totals">
  <div><span class="n">{totals["segments"]}</span><span class="l">segments kept</span></div>
  <div><span class="n">{totals["duration"]:.0f}s</span><span class="l">total length</span></div>
  <div><span class="n">{totals["coverage_pct"]:.1f}%</span><span class="l">of the source</span></div>
</div>
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
