"""Work out why a highlight disappointed, from the record of how it was made.

The report already says what happened. What a dissatisfied user needs is the
next sentence — *why* it happened and which knob changes it — and that is
mostly not a reasoning problem. "Every point came from one signal because the
other six are weighted zero" is a fact about a settings table, derivable from
the record with arithmetic. Findings are therefore computed here,
deterministically, and can be shown with no model present at all.

A language model is useful one layer up: phrasing these findings for someone who
did not write the pipeline, and answering the follow-up question. It is given
the findings and the matching pages from ``docs/advisor`` as its material — the
knowledge lives in the documentation, not in the weights, so it can be corrected
by editing a file. That also bounds what the advisor can say: a model that
cannot invent a finding cannot invent a recommendation.

Each finding carries the evidence that produced it. An advisor the user cannot
check is one they stop trusting the first time it is wrong.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable, Mapping, Optional, Sequence

# Which setting sets each signal's weight. The report stores the settings the
# run used, so a signal that scored nothing can be told apart from one that was
# never switched on.
SIGNAL_WEIGHTS = {
    "scene": "scene_points",
    "motion_event": "motion_event_points",
    "motion_peak": "motion_peak_points",
    "audio": "audio_peak_points",
    "keyword": "keyword_points",
    "object": "object_points",
    "action": "action_points",
    "face": "face_expression_points",
}

SIGNAL_NAMES = {
    "scene": "scene changes",
    "motion_event": "motion events",
    "motion_peak": "motion peaks",
    "audio": "audio peaks",
    "keyword": "transcript keywords",
    "object": "objects",
    "action": "actions",
    "face": "facial expressions",
}

# A cut drawn from less than this fraction of the video is concentrated enough
# to be worth mentioning — the user may have wanted the whole story.
CONCENTRATED_SPAN = 0.35

# A tag in at least this share of the kept clips makes the highlight repetitive.
DOMINANT_TAG_SHARE = 0.8


@dataclass
class Finding:
    """One thing worth telling the user, with the numbers behind it."""
    id: str
    severity: str                     # "high" | "medium" | "low"
    title: str
    detail: str                       # what the record shows, with figures
    remedy: str                       # what to change
    topic: str                        # page in docs/advisor that explains it
    evidence: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def _kept(report: Mapping) -> list:
    return list(report.get("segments") or [])


def _tags(entry: Mapping) -> list:
    return list(entry.get("objects") or []) + list(entry.get("events") or [])


def signal_totals(report: Mapping) -> dict:
    """Points contributed per signal, summed from the clips if not recorded.

    Reports written before the totals were stored are still worth diagnosing —
    they are exactly the runs someone is dissatisfied with — and the figure is
    recoverable from the per-clip breakdowns.
    """
    totals = report.get("signal_totals")
    if totals:
        return dict(totals)
    recovered: dict = {}
    for entry in _kept(report):
        for key, value in (entry.get("breakdown") or {}).items():
            if value:
                recovered[key] = recovered.get(key, 0.0) + float(value)
    return recovered


def _scoring_signals(report: Mapping) -> dict:
    """Signals that actually contributed, ignoring the position bonuses.

    Being near the start or end of a video is not evidence about content, so
    counting it here would make a one-signal run look like a two-signal one.
    """
    return {k: v for k, v in signal_totals(report).items()
            if k in SIGNAL_WEIGHTS and v}


# --------------------------------------------------------------------------- #
# Rules
# --------------------------------------------------------------------------- #

def _rule_single_signal(report) -> Optional[Finding]:
    scoring = _scoring_signals(report)
    if len(scoring) != 1:
        return None
    key, total = next(iter(scoring.items()))
    settings = report.get("settings") or {}
    off = [SIGNAL_NAMES[k] for k, w in SIGNAL_WEIGHTS.items()
           if k != key and not settings.get(w)]
    return Finding(
        id="single_signal",
        severity="high",
        title="Only one kind of evidence decided this highlight",
        detail=(f"All {total:.0f} points came from {SIGNAL_NAMES[key]}. "
                + (f"The other signals ({', '.join(off)}) are weighted at zero, "
                   "so nothing else could influence which moments were chosen."
                   if off else
                   "No other signal scored anything.")),
        remedy=("Give one or two more signals a non-zero weight and re-run. "
                "Moments where several signals agree then rise above moments "
                "where only one fired."),
        topic="weights",
        evidence={"signal": key, "points": total, "disabled": off},
    )


def _rule_silent_detector(report) -> list:
    """Weighted above zero, and yet it never contributed anything."""
    settings = report.get("settings") or {}
    totals = signal_totals(report)
    out = []
    for key, weight_key in SIGNAL_WEIGHTS.items():
        weight = settings.get(weight_key) or 0
        if weight > 0 and not totals.get(key):
            out.append(Finding(
                id=f"silent_{key}",
                severity="medium",
                title=f"{SIGNAL_NAMES[key].capitalize()} are switched on but never fired",
                detail=(f"{SIGNAL_NAMES[key].capitalize()} are worth "
                        f"{weight} points, yet contributed nothing to any kept "
                        "moment. The detector either found nothing or found it "
                        "below its confidence threshold."),
                remedy=("Check that this detector is actually running and that "
                        "its threshold is not set above what your footage "
                        "produces. If it genuinely has no class for what you "
                        "are looking for, teach a category from examples "
                        "instead of raising the weight."),
                topic="thresholds",
                evidence={"signal": key, "weight": weight},
            ))
    return out


def _rule_flat_score(report) -> Optional[Finding]:
    kept = _kept(report)
    if len(kept) < 3:
        return None
    scores = [e["score"] for e in kept]
    if max(scores) != min(scores):
        return None
    return Finding(
        id="flat_score",
        severity="medium",
        title="Every chosen moment scored exactly the same",
        detail=(f"All {len(kept)} clips scored {scores[0]:.0f} points. When "
                "nothing separates the candidates, the order they were picked "
                "in is effectively arbitrary — the ranking had nothing to rank."),
        remedy=("Add a signal that varies across the video, or weight the "
                "existing ones differently so that agreement between them "
                "produces a range of scores instead of one value."),
        topic="weights",
        evidence={"score": scores[0], "clips": len(kept)},
    )


def _rule_concentrated(report) -> Optional[Finding]:
    kept = _kept(report)
    duration = (report.get("video") or {}).get("duration") or 0
    if len(kept) < 2 or duration <= 0:
        return None
    span = max(e["end"] for e in kept) - min(e["start"] for e in kept)
    fraction = span / duration
    if fraction > CONCENTRATED_SPAN:
        return None
    coverage = float((report.get("settings") or {}).get("coverage") or 0.0)
    if coverage >= 0.5:
        return None
    return Finding(
        id="concentrated",
        severity="medium",
        title="The whole highlight came from one stretch of the video",
        detail=(f"Every clip falls inside {span / 60:.0f} minutes of a "
                f"{duration / 60:.0f}-minute video — {fraction * 100:.0f}% of "
                "it. The rest was analysed but never represented, because the "
                "highest-scoring moments happen to sit together."),
        remedy=("Move the Best moments <-> Full story slider to the right. It "
                "caps how much any one part of the video may contribute, so "
                "the cut spans the whole thing without getting any shorter."),
        topic="coverage",
        evidence={"span_seconds": span, "fraction": fraction,
                  "coverage_setting": coverage},
    )


def _rule_boost_never_fired(report) -> Optional[Finding]:
    settings = report.get("settings") or {}
    multiplier = float(settings.get("multi_signal_boost") or 1.0)
    minimum = int(settings.get("min_signals_for_boost") or 0)
    kept = _kept(report)
    if multiplier <= 1.0 or minimum <= 0 or not kept:
        return None
    if any((e.get("boost") or {}).get("applied") for e in kept):
        return None
    best = max((e.get("boost") or {}).get("signal_count", 0) for e in kept)
    return Finding(
        id="boost_never_fired",
        severity="low",
        title="The multi-signal boost never applied",
        detail=(f"The boost rewards moments where at least {minimum} signals "
                f"agree, but no kept moment had more than {best}. It is "
                "configured and has no effect."),
        remedy=(f"Either switch on another detector so signals can coincide, "
                f"or lower the agreement requirement below {minimum}."),
        topic="weights",
        evidence={"multiplier": multiplier, "required": minimum,
                  "best_seen": best},
    )


def _rule_near_miss_gap(report) -> Optional[Finding]:
    """Signals that only ever appear in moments that did not make the cut."""
    kept = _kept(report)
    misses = list(report.get("near_misses") or [])
    if not kept or not misses:
        return None
    in_kept = {s for e in kept for s in (e.get("signals_present") or [])}
    in_missed = {s for e in misses for s in (e.get("signals_present") or [])}
    only_missed = sorted(
        (in_missed - in_kept) & set(SIGNAL_WEIGHTS), key=str
    )
    if not only_missed:
        return None
    names = ", ".join(SIGNAL_NAMES[s] for s in only_missed)
    return Finding(
        id="near_miss_gap",
        severity="medium",
        title="Some evidence only ever appears in moments you did not get",
        detail=(f"Moments involving {names} scored well enough to be listed as "
                "near misses, but none of them made the cut. That signal is "
                "being outvoted everywhere it fires."),
        remedy=(f"If those are moments you wanted, raise the weight of {names} "
                "and re-run. The near-miss table lists the exact timestamps to "
                "check first."),
        topic="weights",
        evidence={"signals": only_missed, "near_misses": len(misses)},
    )


def _rule_dominant_tag(report) -> Optional[Finding]:
    kept = _kept(report)
    if len(kept) < 3:
        return None
    counts: dict = {}
    for entry in kept:
        for tag in set(_tags(entry)):
            counts[tag] = counts.get(tag, 0) + 1
    if not counts:
        return None
    tag, count = max(counts.items(), key=lambda kv: kv[1])
    share = count / len(kept)
    if share < DOMINANT_TAG_SHARE:
        return None
    return Finding(
        id="dominant_tag",
        severity="low",
        title="Nearly every clip contains the same thing",
        detail=(f"'{tag}' appears in {count} of {len(kept)} clips "
                f"({share * 100:.0f}%). The highlight is likely to feel "
                "repetitive even though each moment scored well on its own."),
        remedy=("Selection has no notion of variety — it takes the highest "
                "scores, and if they all look alike, so will the cut. Spread "
                "the cut with the coverage slider, or swap individual clips "
                "for the alternatives that lost to them."),
        topic="variety",
        evidence={"tag": tag, "clips_with_tag": count, "clips": len(kept)},
    )


def _entries_overlapping(report, ranges: Iterable[Sequence[float]]) -> list:
    out = []
    for entry in _kept(report):
        for start, end in ranges:
            if entry["start"] < end and entry["end"] > start:
                out.append(entry)
                break
    return out


def _rule_rejected_share_a_tag(report, rejected) -> Optional[Finding]:
    """What the moments the user turned down had in common."""
    if not rejected:
        return None
    entries = _entries_overlapping(report, rejected)
    if len(entries) < 2:
        return None
    counts: dict = {}
    for entry in entries:
        for tag in set(_tags(entry)):
            counts[tag] = counts.get(tag, 0) + 1
    if not counts:
        return None
    tag, count = max(counts.items(), key=lambda kv: kv[1])
    if count < 2:
        return None
    return Finding(
        id="rejected_share_a_tag",
        severity="high",
        title="The clips you rejected have something in common",
        detail=(f"'{tag}' appears in {count} of the {len(entries)} moments you "
                "swapped away. Whatever is scoring those moments is finding "
                "something you do not want."),
        remedy=(f"Lower the weight of whatever contributes '{tag}', or mark it "
                "as an example of what not to match so future runs score it "
                "down instead of up."),
        topic="weights",
        evidence={"tag": tag, "rejected_with_tag": count,
                  "rejected": len(entries)},
    )


def _rule_short_of_target(report) -> Optional[Finding]:
    settings = report.get("settings") or {}
    target = float(settings.get("target_duration") or 0)
    got = float((report.get("totals") or {}).get("duration") or 0)
    if target <= 0 or got >= target * 0.8:
        return None
    return Finding(
        id="short_of_target",
        severity="medium",
        title="The highlight came out shorter than you asked for",
        detail=(f"You asked for up to {target:.0f}s and got {got:.0f}s. In MAX "
                "mode the cut stops when it runs out of moments that scored "
                "anything at all — not when it runs out of budget."),
        remedy=("Lower the detector thresholds so more moments score, give "
                "another signal a weight, or accept the shorter cut: padding it "
                "means including moments nothing was detected in."),
        topic="thresholds",
        evidence={"target": target, "actual": got},
    )


def _rule_expressions_unselected(report) -> Optional[Finding]:
    """Weighted, scanned, and scoring nothing because no class was chosen."""
    settings = report.get("settings") or {}
    weight = settings.get("face_expression_points") or 0
    labels = settings.get("face_expression_labels") or []
    if weight <= 0 or labels:
        return None
    return Finding(
        id="expressions_unselected",
        severity="medium",
        title="Expressions are weighted but no expression was chosen",
        detail=(f"Facial expressions are worth {weight} points, but no class "
                "is selected, so nothing can match. The scan does not even "
                "run — there is no outcome it could change."),
        remedy=("Name the expressions that matter to you. Matching all of them "
                "would score every second a face is visible, which separates "
                "nothing, so the choice is deliberately yours."),
        topic="weights",
        evidence={"weight": weight},
    )


def _rule_expression_lopsided(report) -> Optional[Finding]:
    """One class swallowing the scan usually means the model cannot read these
    faces at all — the failure it is most important to name, because it looks
    exactly like a confident answer."""
    counts = (report.get("settings") or {}).get("face_expression_counts") or {}
    total = sum(int(v) for v in counts.values())
    if total < 30:
        return None
    label, count = max(counts.items(), key=lambda kv: int(kv[1]))
    share = int(count) / total
    if share < 0.45:
        return None
    return Finding(
        id="expression_lopsided",
        severity="medium",
        title="One expression accounts for most of the video",
        detail=(f"'{label}' was reported for {count} of {total} readable "
                f"seconds ({share * 100:.0f}%). The classifier was trained on "
                "posed, frontal faces and returns a label for every face it is "
                "given, so on footage unlike that training set a single class "
                "dominating usually means it cannot read these faces — not "
                "that the video is mostly that expression."),
        remedy=("Jump to a few of those moments and check the label against "
                "what is on screen. If it does not hold up, the built-in "
                "classes are the wrong tool for this footage and a category "
                "taught from your own example crops is the answer."),
        topic="training",
        evidence={"label": label, "count": int(count), "readable": total,
                  "share": round(share, 3)},
    )


def diagnose(report: Mapping,
             *,
             rejected: Optional[Sequence[Sequence[float]]] = None) -> list:
    """Every finding the record supports, most serious first.

    ``rejected`` is the time ranges the user swapped away, which turns "here is
    what your settings do" into "here is what your settings do that you did not
    want".
    """
    findings = []
    for rule in (_rule_single_signal, _rule_flat_score, _rule_concentrated,
                 _rule_boost_never_fired, _rule_near_miss_gap,
                 _rule_dominant_tag, _rule_short_of_target,
                 _rule_expressions_unselected, _rule_expression_lopsided):
        found = rule(report)
        if found:
            findings.append(found)
    findings.extend(_rule_silent_detector(report))
    rejected_finding = _rule_rejected_share_a_tag(report, rejected)
    if rejected_finding:
        findings.append(rejected_finding)

    order = {"high": 0, "medium": 1, "low": 2}
    findings.sort(key=lambda f: order.get(f.severity, 3))
    return findings


def attach_advice(report: dict,
                  *,
                  rejected: Optional[Sequence[Sequence[float]]] = None) -> dict:
    """Put the findings into the record, so page and model read the same list."""
    report["advice"] = [f.as_dict() for f in diagnose(report, rejected=rejected)]
    return report
