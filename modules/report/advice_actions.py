"""Turn the advice the user ticked into one re-run's settings.

``highlight_advice`` attaches an ``action`` to every finding whose remedy is a
concrete settings change. Several of them are usually worth taking together, and
running once per finding would pay for the run several times over. So the
chosen actions are merged here into one change, applied on top of the settings
the report says the run used — the run being improved, not whatever the window
happens to hold now.

Two things are said up front, before anything runs:

* **Conflicts.** Two findings can move the same setting to different values.
  The stronger change wins (the higher weight or coverage; for the boost's
  agreement requirement, the lower bar), and the conflict is reported rather
  than resolved silently.
* **Cost.** Changing a weight re-scores from the cache in seconds. Turning on a
  signal whose detector was skipped (every weight that gates it was zero) means
  running that detector first, which takes minutes. ``analysis_plan`` already
  knows which settings gate which detector, so the answer comes from there.

No Qt and no I/O: everything here is a function of two dicts.
"""
from __future__ import annotations

from typing import Iterable, Mapping, Sequence

from modules.report.analysis_plan import GATED_ARTIFACTS, gate_is_open
from modules.report.highlight_advice import SIGNAL_WEIGHTS

# The run settings a report records and a re-run takes back from it, keyed as
# the pipeline reads them.
RERUN_KEYS = tuple(SIGNAL_WEIGHTS.values()) + (
    "loudness_burst_points",
    "face_expression_labels",
    "multi_signal_boost",
    "min_signals_for_boost",
    "coverage",
    "clip_time",
)

# Settings where the smaller value is the stronger change.
LOWER_IS_STRONGER = {"min_signals_for_boost"}

RESCORE, ANALYSE = "rescore", "analyse"


def recorded_settings(report: Mapping) -> dict:
    """The settings the report's run used, as run settings again."""
    recorded = dict(report.get("settings") or {})
    out = {k: recorded[k] for k in RERUN_KEYS if k in recorded}
    target = recorded.get("target_duration")
    if target:
        if recorded.get("duration_mode") == "EXACT":
            out["exact_duration"] = target
        else:
            out["max_duration"] = target
    return out


def actionable(findings: Iterable[Mapping]) -> list:
    """The findings (as stored in a report's ``advice``) that carry a change."""
    return [f for f in findings
            if (f.get("action") or {}).get("settings")]


def merge(actions: Sequence[Mapping]) -> tuple[dict, list]:
    """``(change, conflicts)`` for several actions applied at once.

    ``conflicts`` lists ``{"setting", "values", "chosen"}`` for every setting two
    actions disagree on."""
    proposed: dict = {}
    for action in actions:
        for key, value in (action.get("settings") or {}).items():
            proposed.setdefault(key, [])
            if value not in proposed[key]:
                proposed[key].append(value)
    change, conflicts = {}, []
    for key, values in proposed.items():
        chosen = (min(values) if key in LOWER_IS_STRONGER else max(values))
        change[key] = chosen
        if len(values) > 1:
            conflicts.append({"setting": key, "values": values, "chosen": chosen})
    return change, conflicts


def plan_rerun(report: Mapping, chosen: Sequence[Mapping]) -> dict:
    """Everything a window needs to show before running the chosen findings.

    ``chosen`` are finding dicts from the report's ``advice``. Returns the full
    settings for the re-run, the change on its own, conflicts, the detectors
    that must run first, the cost, and one line per changed setting."""
    base = recorded_settings(report)
    change, conflicts = merge([f.get("action") or {} for f in chosen])
    after = {**base, **change}
    analyses = [name for name in GATED_ARTIFACTS
                if not gate_is_open(base, name) and gate_is_open(after, name)]
    lines = []
    for key, value in change.items():
        before = base.get(key)
        if before != value:
            lines.append(f"{key}: {_fmt(before)} → {_fmt(value)}")
    return {
        "settings": after,
        "change": change,
        "conflicts": conflicts,
        "analyses": analyses,
        "cost": ANALYSE if analyses else RESCORE,
        "lines": lines,
        "chosen": [f.get("id") for f in chosen],
    }


def _fmt(value) -> str:
    if value is None:
        return "unset"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)
