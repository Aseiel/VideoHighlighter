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


# What a user can say is wrong, and the findings each complaint makes relevant.
# Naming the complaint is what turns "add another signal" into "this one".
CONCERNS = {
    "repetitive": "the clips are too similar to each other",
    "wrong_moments": "it picked moments I did not want",
    "missed_moments": "it missed moments I did want",
    "too_short": "the highlight is shorter than I asked for",
    "arbitrary": "the choices look random",
}


def _rule_unused_detector(report, concern=None) -> list:
    """A detector that found plenty and was never allowed to count.

    Scenes, motion and audio peaks are detected whatever their weight, so when
    the record says a detector fired hundreds of times at zero points, the
    advice can name it instead of suggesting "a signal" in the abstract.
    """
    settings = report.get("settings") or {}
    activity = settings.get("detector_activity") or {}
    kept = len(_kept(report))
    out = []
    for key, weight_key in SIGNAL_WEIGHTS.items():
        found = int(activity.get(key) or 0)
        weight = settings.get(weight_key) or 0
        # Enough to discriminate: comfortably more events than clips, or it
        # would only re-rank the moments already chosen.
        if weight > 0 or found < max(20, kept * 3):
            continue
        out.append(Finding(
            id=f"unused_{key}",
            severity="high" if concern in ("arbitrary", "repetitive") else "medium",
            title=f"{SIGNAL_NAMES[key].capitalize()} were found but count for nothing",
            detail=(f"{found} {SIGNAL_NAMES[key]} were detected across this "
                    f"video and are weighted at zero, so none of them could "
                    f"influence the {kept} moments that were chosen."),
            remedy=(f"Give {SIGNAL_NAMES[key]} a small weight — 3 to 5 points "
                    "is usually enough to break ties without taking over — and "
                    "re-run. Detection is cached, so it costs a re-score, not "
                    "a re-analysis."),
            topic="weights",
            evidence={"signal": key, "detected": found, "clips": kept},
        ))
    # Most events first: the detector with the most to say is the best bet.
    out.sort(key=lambda f: -f.evidence["detected"])
    return out


def _rule_concern_repetitive(report, concern=None) -> Optional[Finding]:
    """Asked for variety when nothing in the scoring rewards it."""
    if concern != "repetitive":
        return None
    scoring = _scoring_signals(report)
    if len(scoring) > 1:
        return None
    key = next(iter(scoring), "")
    return Finding(
        id="concern_repetitive",
        severity="high",
        title="Nothing in this run could have produced variety",
        detail=(f"Every point came from {SIGNAL_NAMES.get(key, key)}, so the "
                "highest-scoring moments are the ones where that signal was "
                "strongest — and those tend to look alike. Selection has no "
                "notion of variety; it takes the top scores."),
        remedy=("Two things help, in this order: raise the Best moments <-> "
                "Full story slider, which spreads the cut across the video, "
                "and give a second signal a weight so a moment has to be "
                "interesting in more than one way to win."),
        topic="variety",
        evidence={"signal": key},
    )


def _rule_unchecked_vocabulary(report) -> Optional[Finding]:
    """The video talks about things no class covers, so no signal can check them.

    The most consequential finding this module can make once a transcript is
    involved, because it is the only one that is about the *limits* of the run
    rather than its settings. Everything else here says a weight is wrong; this
    says the run could not have answered the question, however the weights were
    set — and names what would have to exist before it could.
    """
    vocabulary = report.get("vocabulary") or {}
    gaps = vocabulary.get("gaps") or []
    if not gaps:
        return None
    words = ", ".join(str(g["word"]) for g in gaps[:4])
    classes = vocabulary.get("classes") or []
    lead = gaps[0]
    where = (lead.get("chapters") or [{}])[0]
    return Finding(
        id="unchecked_vocabulary",
        severity="medium",
        title="The video talks about things nothing was watching for",
        detail=(f"{len(gaps)} word(s) are said far more often in one stretch "
                f"than in the rest of the video and match no class this "
                f"detector produced here ({words}). The strongest is "
                f"\"{lead['word']}\", said {lead['count']} time(s) at "
                f"{lead['times']:.0f}× the video's rate, around "
                f"{where.get('timestamp', 'a chapter')}. The detector produced "
                f"{len(classes)} class(es) in this file, so any claim resting "
                f"on those words was neither confirmed nor contradicted — it "
                f"was simply not measured."),
        remedy=("Decide whether any of those words describes an arrangement of "
                "classes you already detect. If it does, a composition rule "
                "will label it and the next run can check the claim; if it does "
                "not, the detector has no way to see it and no setting will "
                "change that. The advisor can draft the rule for you."),
        topic="composition",
        evidence={"gaps": gaps[:4], "classes": classes},
    )


def _rule_unmeasured_claim(report) -> Optional[Finding]:
    """A sentence the report prints, about something the run has no signal for.

    The companion to :func:`_rule_unchecked_vocabulary` and not a duplicate of
    it. That one is about *words a stretch repeats*, and its answer is a
    composition rule over classes that already exist. This one is about a single
    substantial line — the kind of thing somebody says once, naming what they
    wanted or what they liked most — and its answer is usually not a rule at
    all, because there is nothing to compose. It is a new signal or nothing, and
    the useful part of the advice is which signal, at what cost.

    Deliberately names two routes and not six. The routes themselves come from
    :mod:`modules.detection_routes`, which knows what this run has the
    prerequisites for; what the claim is *about* decides which of the two fits,
    and only the person who can see their own footage knows that.
    """
    unmeasured = report.get("unmeasured") or {}
    claims = unmeasured.get("claims") or []
    routes = unmeasured.get("routes") or {}
    if not claims:
        return None

    lead = claims[0]
    where = lead.get("chapter") or {}
    watched = lead.get("measured_here") or []
    # The two cases read very differently to a user. "Nothing was detected in
    # that stretch" is a run that did not look; "these four classes were, and
    # none of them is this" is a run that looked and has no vocabulary for it,
    # which is the case no threshold or weight can improve.
    context = (f"The detector was busy in that stretch — it labelled "
               f"{', '.join(str(c) for c in watched[:4])} there — so this is "
               f"not a gap in coverage but a gap in vocabulary."
               if watched else
               "Nothing at all was detected in that stretch, so it is worth "
               "checking whether the detector ran there before concluding "
               "anything about what it cannot see.")

    fastest = routes.get("fastest") or {}
    strongest = routes.get("strongest") or {}
    first = routes.get("first") or {}
    steps = []
    if first:
        steps.append("First, for nothing: ask the advisor to draft a rule for "
                     "the line. It answers that the claim cannot be built from "
                     "the classes this video has, when it cannot, and that "
                     "rules out the cheapest route in one call.")
    if fastest:
        steps.append(f"Fastest that would actually measure it: "
                     f"{fastest['name'].lower()} — {fastest['effort']}. "
                     f"Right route when {fastest['holds_when']}; not when "
                     f"{fastest['fails_when']}.")
    if strongest:
        steps.append(f"Most reliable: {strongest['name'].lower()} — "
                     f"{strongest['effort']}. "
                     f"Worth it when {strongest['holds_when']}.")
    probe = routes.get("probe") or {}
    if probe:
        steps.append(probe["why"])

    others = len(claims) - 1
    return Finding(
        id="unmeasured_claim",
        severity="medium",
        title="Something was said that this run has no measurement for",
        detail=(f"At {lead.get('timestamp', '?')}, in chapter "
                f"{where.get('number', '?')}: \"{lead.get('quote', '')}\" "
                f"No class or event this run produced shares a word with that "
                f"line, so the report quotes it and says nothing about it — "
                f"neither confirming nor contradicting it. {context}"
                + (f" {others} other quoted line(s) are in the same position."
                   if others > 0 else "")),
        remedy=" ".join(steps) or (
            "This run produced no detections at all, so there is nothing to "
            "build on yet. Run object detection first, then ask again."),
        topic="measuring",
        evidence={"claims": claims[:3], "routes": routes},
    )


def _rule_pending_checks(report) -> list:
    """An earlier run added a rule to test a claim. Say how it turned out."""
    out = []
    for check in (report.get("checks") or []):
        state = str(check.get("state") or "")
        rule = str(check.get("rule") or "")
        claim = str(check.get("claim") or "").strip()
        quoted = f' The claim was: "{claim}"' if claim else ""
        if state == "fired":
            out.append(Finding(
                id=f"check_fired:{rule}",
                severity="low",
                title=f"The rule added to check a claim fired: {rule}",
                detail=(f"'{rule}' matched {int(check.get('seconds') or 0)} "
                        f"second(s), between {format_seconds(check.get('first'))} "
                        f"and {format_seconds(check.get('last'))}.{quoted} "
                        "That the arrangement occurred is measured; what it "
                        "means for the claim is still a reading."),
                remedy=("Watch the seconds it marked before relying on it. A "
                        "rule can fire on the right arrangement of the wrong "
                        "boxes, and this one has not been checked by eye yet."),
                topic="composition",
                evidence={"check": check},
            ))
        elif state == "never_fired":
            out.append(Finding(
                id=f"check_silent:{rule}",
                severity="medium",
                title=f"The rule added to check a claim never fired: {rule}",
                detail=(f"'{rule}' is in the rules and matched nothing in this "
                        f"run.{quoted} That is weak evidence against the claim "
                        "and not proof: the arrangement may have happened "
                        "outside the detector's view, or the rule may be too "
                        "strict for it."),
                remedy=("Loosen the rule before concluding anything — widen the "
                        "count range or the time window and re-run. If it still "
                        "finds nothing, the claim is unsupported by what this "
                        "detector can see."),
                topic="composition",
                evidence={"check": check},
            ))
        elif state == "not_in_rules":
            out.append(Finding(
                id=f"check_missing:{rule}",
                severity="high",
                title=f"A claim is waiting on a rule that is not loaded: {rule}",
                detail=(f"An earlier run recorded '{rule}' as the way to check "
                        f"a claim, and the composition engine is not evaluating "
                        f"it.{quoted} Nothing in this report says anything about "
                        "that claim either way."),
                remedy=("Check the rule is still in your composition rules file "
                        "and that the object detection pass actually re-ran — a "
                        "cached detection pass skips the composition engine "
                        "entirely."),
                topic="composition",
                evidence={"check": check},
            ))
    return out


def format_seconds(value) -> str:
    """``h:mm:ss`` for a finding's detail line. Local to keep this module free
    of a report import for one string."""
    try:
        seconds = max(0, int(value))
    except (TypeError, ValueError):
        return "?"
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def diagnose(report: Mapping,
             *,
             rejected: Optional[Sequence[Sequence[float]]] = None,
             concern: Optional[str] = None) -> list:
    """Every finding the record supports, most serious first.

    ``rejected`` is the time ranges the user swapped away, which turns "here is
    what your settings do" into "here is what your settings do that you did not
    want".

    ``concern`` is what the user said was wrong (see :data:`CONCERNS`). Findings
    are the same facts either way; naming the complaint decides which of them
    lead, and adds the ones that only make sense as an answer to a question. An
    advisor that does not know what disappointed you can only list everything
    it noticed.
    """
    findings = []
    for rule in (_rule_single_signal, _rule_flat_score, _rule_concentrated,
                 _rule_boost_never_fired, _rule_near_miss_gap,
                 _rule_dominant_tag, _rule_short_of_target,
                 _rule_expressions_unselected, _rule_expression_lopsided,
                 _rule_unchecked_vocabulary, _rule_unmeasured_claim):
        found = rule(report)
        if found:
            findings.append(found)
    findings.extend(_rule_pending_checks(report))
    findings.extend(_rule_silent_detector(report))
    findings.extend(_rule_unused_detector(report, concern))
    concern_finding = _rule_concern_repetitive(report, concern)
    if concern_finding:
        findings.append(concern_finding)
    rejected_finding = _rule_rejected_share_a_tag(report, rejected)
    if rejected_finding:
        findings.append(rejected_finding)

    order = {"high": 0, "medium": 1, "low": 2}
    findings.sort(key=lambda f: order.get(f.severity, 3))
    return findings


def attach_advice(report: dict,
                  *,
                  rejected: Optional[Sequence[Sequence[float]]] = None,
                  concern: Optional[str] = None) -> dict:
    """Put the findings into the record, so page and model read the same list.

    Derives the unmeasured-claims section first when the record predates it.
    Every path that diagnoses a run goes through here, and a finding that could
    only fire on a freshly analysed video would be missing from precisely the
    reports somebody came back to because they were unhappy with them.
    """
    try:
        from modules.uncovered_claims import ensure
        ensure(report)
    except Exception as exc:                       # pragma: no cover - defensive
        print(f"⚠️ Unmeasured claims skipped: {exc}")
    report["advice"] = [f.as_dict() for f in
                        diagnose(report, rejected=rejected, concern=concern)]
    if concern:
        report["advice_concern"] = str(concern)
    return report
