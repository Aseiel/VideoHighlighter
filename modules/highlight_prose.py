"""Say what a moment was, in a sentence, from what was measured.

The report already knows a clip peaked at -1 dBFS in the 100th percentile with
two signals landing 0.3s apart. A person reading that has to do the translation
themselves, every row, and mostly does not — so a report full of true numbers
communicates less than one plain sentence would.

This is the translation, and it is deterministic. Every clause is produced by a
threshold over a measurement, so a sentence can be traced back to the figures
printed beside it and cannot say more than the data supports. No model is
involved: a language model asked to do this would produce better prose and
occasionally invent a reason, and the second thing costs more than the first
gains.

The hardest discipline here is refusing to flatter. Most moments in most videos
are not exceptional, and a report that calls every one of them outstanding is
worth exactly as much as one that says nothing. When the numbers are ordinary
the sentence says so — and when everything in a run scored alike, saying *that*
is the most useful sentence available, because it means the ranking had nothing
to work with.
"""
from __future__ import annotations

import math
from typing import Mapping, Optional, Sequence

# Percentile at which a moment is genuinely unusual for its video rather than
# merely above average. Set high on purpose: the word has to keep its meaning.
EXCEPTIONAL = 90.0
NOTABLE = 70.0
# Wide on purpose. Most clips in most cuts are unremarkable relative to each
# other, and a narrow middle band pushes ordinary clips into "weaker", which
# reads as a criticism of a clip that is doing nothing wrong.
ORDINARY = 30.0

# A score shared by at least this much of the cut is a tie, not a rank. A
# majority, because the sentence says "most" and it has to be true.
TIED_SHARE = 0.5

# Loudness is judged against the file, not an absolute level — a quiet recording
# has loud moments too, and they are what its highlights are made of.
LOUD_PERCENTILE = 90.0

# A class carried by this share of the video's detected seconds or less is rare
# enough that its presence is itself the reason a moment stands out. Above it,
# saying "this clip has one" describes the whole file, not the clip.
RARE_PREVALENCE = 12.0

# ...but a class seen fewer times than this is not rare, it is unconfirmed, and
# the two read identically in a sentence. This is the only gate rarity gets:
# holding it to the sample count the *size* comparison needs would be circular,
# because a class rare enough to be worth mentioning can never clear it.
MIN_RARE_DETECTIONS = 5

# A subject present for less than this much of a clip is a flicker. A size
# percentile computed off one such frame is arithmetically fine and rhetorically
# a lie, so the share goes in the sentence rather than being quietly dropped —
# the reader is the one who should decide whether to believe it.
FLEETING_PRESENCE = 25.0

# How far a size ratio has to sit from the video's own median before it is worth
# a sentence, as a fraction of that median. A percentile answers "is this the
# largest" and says nothing about "by how much" — and on classes whose boxes
# overlap, the largest is routinely 8% across and still lands in the 98th, which
# is a true number and a worthless sentence. Effect size is the missing gate.
MIN_RATIO_EFFECT = 0.25

# How much more of a clip an expression has to be, relative to its share of the
# whole video, before the difference is worth a sentence. Twice is the point at
# which "more here than elsewhere" stops being sampling noise across the handful
# of seconds a clip contains.
EXPRESSION_LIFT = 2.0

# Names as a reader would say them, not as the weight table spells them.
SIGNAL_PROSE = {
    "scene": "a shot change",
    "motion_event": "movement",
    "motion_peak": "a burst of movement",
    "audio": "a rise in sound",
    "keyword": "something said",
    "object": "what was on screen",
    "action": "a recognised action",
    "face": "a facial expression",
}


def _join(items: Sequence[str]) -> str:
    items = [i for i in items if i]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def _standing(entry: Mapping, peer_scores: Optional[Sequence[float]]) -> str:
    """How this clip compares with the others in the same cut.

    Deliberately *not* its percentile against the whole video. Selection picks
    the highest-scoring seconds, so a kept clip is in the top percentile of its
    own file by construction — saying so is true, tautological, and reads as
    praise. What a reader cannot already infer is whether this clip is the
    strongest of the ones they got, or the one that scraped in.
    """
    if not peer_scores or len(peer_scores) < 3:
        return ""
    score = float(entry.get("score") or 0.0)
    ordered = sorted(peer_scores)
    if ordered[0] == ordered[-1]:
        return "Scored the same as every other clip here"

    # A clip tied with most of the cut has no rank worth reporting. Ranking it
    # anyway lands the whole tied group mid-scale and then calls them all
    # weaker — which is how fifteen of sixteen top-scoring clips came to be
    # described as the ones that scraped in. A tie is information about the
    # scoring, not about the clip, so say that instead.
    tied = sum(1 for s in ordered if s == score)
    if tied / len(ordered) >= TIED_SHARE:
        return "Scored the same as most of the other clips here"

    # Midrank, as elsewhere: counting only what is strictly below caps the best
    # of five clips at 80%, so the top of a short cut could never be called the
    # top of it.
    below = sum(1 for s in ordered if s < score)
    at_or_below = sum(1 for s in ordered if s <= score)
    share = (below + at_or_below) / 2.0 / len(ordered) * 100.0
    if share >= EXCEPTIONAL:
        return "The strongest clip in this highlight"
    if share >= NOTABLE:
        return "Among the stronger clips here"
    if share >= ORDINARY:
        return "Middling for this highlight"
    return "One of the weaker clips that still made the cut"


def _video_share(measured: Mapping) -> str:
    """How much of the video this moment outscored.

    Kept in the sentence rather than only in the figures below it: it is the
    one number that answers "was this worth keeping" without knowing anything
    about the weight table, and it is the first thing people look for.

    It is honest in both directions. A high share means the scoring was
    selective; a low one means much of the video scored comparably, and the
    choice between those moments was closer to arbitrary than it looks.
    """
    percentile = measured.get("score_percentile")
    if percentile is None:
        return ""
    return f"outscored {percentile:.0f}% of the video"


def _evidence(entry: Mapping) -> str:
    """What actually fired, named the way a person would name it.

    Taken from ``signals_present`` rather than the points, because a signal can
    contribute without scoring — an action suppressed by "require objects" was
    still detected, and a sentence that omits it describes the wrong moment.
    Points are the fallback for a record that predates the field.
    """
    present = set(entry.get("signals_present") or ())
    if not present:
        present = {key for key, value in (entry.get("breakdown") or {}).items()
                   if value > 0}
    ordered = [key for key in SIGNAL_PROSE if key in present]
    return _join([SIGNAL_PROSE[key] for key in ordered])


def _loudness(measured: Mapping) -> str:
    percentile = measured.get("loudness_percentile")
    dbfs = measured.get("loudness_dbfs")
    if percentile is None or dbfs is None:
        return ""
    if percentile >= LOUD_PERCENTILE:
        return f"one of its loudest points ({dbfs:.0f} dBFS)"
    return ""


def _confidence(entry: Mapping, measured: Mapping) -> str:
    """How sure the detector was about what it saw.

    A moment can score well on a detection the model was barely willing to
    make. The score does not carry that — points are the same whether the
    detector was certain or guessing — so without this a reader cannot tell a
    solid pick from one resting on a 0.31.

    Actions are preferred when present because they carry their own confidence
    and a tier; otherwise the strongest box at the peak second.
    """
    actions = entry.get("actions") or []
    if actions:
        best = max(actions, key=lambda a: float(a.get("confidence") or 0.0))
        tier = best.get("tier")
        name = str(best.get("name") or "an action")
        return (f"{name} recognised at {float(best['confidence']):.2f}"
                + (f" ({tier})" if tier else ""))
    confidence = measured.get("detection_confidence")
    if confidence is None:
        return ""
    # The strongest box at that second, not the only one — say so, or a
    # clip with one certain detection and three doubtful ones reads as
    # uniformly certain.
    return f"its strongest detection at {float(confidence):.2f}"


def _agreement(entry: Mapping, measured: Mapping) -> str:
    """Signals arriving together is the difference between noise and an event.

    Phrased to follow the list of what fired, so the reader learns *which*
    signals agreed rather than only how many.
    """
    if len(entry.get("signals_present") or []) < 2:
        return ""
    if not measured.get("signals_coincide"):
        return "though not at the same instant"
    spread = measured.get("signal_spread_seconds", 0.0)
    if spread <= 0:
        return "landing on the same second"
    return f"landing within {spread:.0f}s of each other"


def describe(entry: Mapping,
             peer_scores: Optional[Sequence[float]] = None) -> str:
    """One sentence about why this clip is in the highlight.

    Every clause comes from a threshold over a measurement in ``entry``, so the
    sentence and the figures printed beside it can never disagree.

    ``peer_scores`` are the scores of the other kept clips; without them the
    sentence simply omits any ranking rather than falling back to the
    video-wide percentile, which is tautological for a selected clip.
    """
    measured = entry.get("measured") or {}
    standing = _standing(entry, peer_scores)
    evidence = _evidence(entry)
    agreement = _agreement(entry, measured)

    parts = []
    share = _video_share(measured)
    if share:
        parts.append(share)
    if evidence:
        # What fired always leads the evidence: "two signals agreed" is worth
        # much less than knowing it was sound and movement.
        only = ("" if len(entry.get("signals_present") or []) != 1
                else " alone")
        parts.append(f"{evidence}{only} {agreement}".strip()
                     if agreement else f"{evidence}{only}")
    confidence = _confidence(entry, measured)
    if confidence:
        parts.append(confidence)
    loudness = _loudness(measured)
    if loudness:
        parts.append(loudness)

    if standing:
        return f"{standing} — {_join(parts)}." if parts else f"{standing}."
    if not parts:
        return ""
    # No ranking to lead with, so the first clause becomes the sentence.
    head, rest = parts[0], parts[1:]
    head = head[0].upper() + head[1:]
    return f"{head} — {_join(rest)}." if rest else f"{head}."


# A size claim resting on a box the detector was this unsure of should carry the
# number that undermines it. Same threshold the expression classifier uses for
# "I am picking between near-ties".
WEAK_DETECTION = 0.5


def _subject_line(subject: Mapping) -> str:
    """What is unusual about one detected class here, or nothing.

    The order of preference is the order of trustworthiness. A ratio against
    something else in the same frame comes first because moving the camera
    scales both boxes and leaves it unchanged; bare frame share comes second and
    carries the caveat that it cannot separate a larger subject from a closer
    camera; rarity comes last and needs no size claim at all.
    """
    name = str(subject.get("name") or "").strip()
    if not name:
        return ""

    relative = subject.get("relative") or {}
    lead = ""
    median = float(relative.get("median") or 0.0)
    ratio = float(relative.get("ratio") or 0.0)
    # Ranked high *and* actually different. Either alone is a sentence that
    # misleads: a big lead over nothing, or a big number that is the norm here.
    big_enough = (median > 0 and abs(ratio / median - 1.0) >= MIN_RATIO_EFFECT)
    if (relative.get("enough_samples") and big_enough
            and float(relative.get("percentile") or 0.0) >= NOTABLE):
        linear = float(relative.get("linear_ratio")
                       or math.sqrt(max(0.0, float(relative["ratio"]))))
        lead = (
            f"{name} covers {float(relative['ratio']):.1f}× the area of the "
            f"{relative['reference']} beside it — about {linear:.1f}× across — "
            f"larger than in {float(relative['percentile']):.0f}% of this "
            f"video's {relative['stretch_seconds']}s stretches where the two "
            f"share a frame, against a usual {float(relative['median']):.1f}×"
        )
    elif (subject.get("enough_samples")
          and float(subject.get("frame_share_percentile") or 0.0) >= NOTABLE):
        lead = (
            f"{name} fills {float(subject['frame_share']):.1f}% of the frame — "
            f"more than in {float(subject['frame_share_percentile']):.0f}% of "
            f"this video's other {subject['stretch_seconds']}s stretches "
            f"containing one, though frame share also rises when the camera "
            f"simply moves closer"
        )

    prevalence = subject.get("prevalence_pct")
    rarity = ""
    if (prevalence is not None and float(prevalence) <= RARE_PREVALENCE
            and int(subject.get("detections") or 0) >= MIN_RARE_DETECTIONS):
        rarity = (f"{name} is in only {float(prevalence):.0f}% of the video's "
                  f"detected seconds")

    if lead and rarity:
        # The rarity clause has already named the class; don't say it twice.
        lead = f"{lead}, and it is a class in only {float(prevalence):.0f}% of " \
               f"the video's detected seconds"
    elif rarity:
        lead = rarity
    if not lead:
        return ""

    caveats = []
    presence = subject.get("clip_presence_pct")
    if presence is not None and float(presence) < FLEETING_PRESENCE:
        caveats.append(f"present for only {float(presence):.0f}% of the clip")
    confidence = subject.get("confidence")
    if confidence is not None and float(confidence) < WEAK_DETECTION:
        caveats.append(f"on a {float(confidence):.2f} detection")
    if caveats:
        lead = f"{lead} — {_join(caveats)}"
    return lead[0].upper() + lead[1:] + "."


def _expression_line(expression: Mapping) -> str:
    """How the expression read here against how it reads elsewhere.

    Phrased throughout as something the *classifier* reported, never as a claim
    about the person on screen. That is not politeness: the model has five coarse
    classes, no notion of intensity, degrades on profile and occlusion, and
    returns a label for every face it is handed. "The classifier read surprise
    more strongly here than anywhere else in the file" is defensible and is
    genuinely what the reader wants; anything stronger is not supported.
    """
    label = str(expression.get("label") or "").strip()
    if not label or not expression.get("enough_samples"):
        return ""

    confidence = float(expression.get("confidence") or 0.0)
    percentile = expression.get("confidence_percentile")
    lift = float(expression.get("lift") or 0.0)
    clip_share = float(expression.get("clip_share_pct") or 0.0)
    video_share = float(expression.get("video_share_pct") or 0.0)

    strong = percentile is not None and float(percentile) >= NOTABLE
    unusual = lift >= EXPRESSION_LIFT
    if not (strong or unusual):
        return ""

    parts = []
    if strong:
        parts.append(
            f"read {label} at {confidence:.2f} — stronger than in "
            f"{float(percentile):.0f}% of this video's other "
            f"{expression['stretch_seconds']}s stretches carrying that label"
        )
    else:
        parts.append(f"read {label} at {confidence:.2f}")
    if unusual:
        parts.append(f"and {label} covers {clip_share:.0f}% of this clip "
                     f"against {video_share:.0f}% of the video ({lift:.1f}×)")

    dominant = expression.get("video_dominant")
    if dominant and dominant != label:
        parts.append(f"a video the classifier otherwise reads as mostly "
                     f"{dominant} ({float(expression.get('video_dominant_share_pct') or 0.0):.0f}%)")

    return "The expression classifier " + ", ".join(parts) + "."


def explain_standout(entry: Mapping) -> list:
    """The deeper reading of one clip: what was different about what was on screen.

    Separate from :func:`describe` on purpose. That sentence is a summary and
    has to stay one line; this is the evidence a reader turns to when they want
    to argue with the pick, and cramming it into the same sentence would cost
    both of them their job.

    Returns an empty list when nothing cleared the thresholds — which is the
    common case, and is the point. Most clips are not unusual in any measurable
    way, and a paragraph of hedged findings under every one of them would train
    the reader to skip the section entirely.
    """
    comparison = (entry.get("measured") or {}).get("comparison") or {}
    lines = []
    for subject in comparison.get("subjects") or ():
        line = _subject_line(subject)
        if line:
            lines.append(line)
    expression = _expression_line(comparison.get("expression") or {})
    if expression:
        lines.append(expression)
    return lines


def summarise_standouts(report: Mapping) -> str:
    """One line naming the clip each comparison axis actually singled out.

    Worth its own sentence because the per-clip findings are scattered down the
    page: a reader who wants "so which one is the unusual one" has to hold nine
    rows in their head to answer it, and this answers it once.
    """
    segments = list(report.get("segments") or [])
    best_subject = None
    best_expression = None
    for entry in segments:
        comparison = (entry.get("measured") or {}).get("comparison") or {}
        for subject in comparison.get("subjects") or ():
            relative = subject.get("relative") or {}
            if not relative.get("enough_samples"):
                continue
            rank = float(relative.get("percentile") or 0.0)
            if rank >= EXCEPTIONAL and (best_subject is None
                                        or rank > best_subject[0]):
                best_subject = (rank, entry, subject)
        expression = comparison.get("expression") or {}
        if expression.get("enough_samples"):
            lift = float(expression.get("lift") or 0.0)
            if lift >= EXPRESSION_LIFT and (best_expression is None
                                            or lift > best_expression[0]):
                best_expression = (lift, entry, expression)

    parts = []
    if best_subject:
        rank, entry, subject = best_subject
        parts.append(
            f"the {subject['name']} is at its largest relative to the "
            f"{subject['relative']['reference']} at "
            f"{entry.get('timestamp', '?')} ({rank:.0f}th percentile for this video)"
        )
    if best_expression:
        lift, entry, expression = best_expression
        parts.append(
            f"the expression classifier reads {expression['label']} furthest "
            f"above its video-wide rate at {entry.get('timestamp', '?')} "
            f"({lift:.1f}×)"
        )
    if not parts:
        return ""
    return ("Across the clips kept, " + _join(parts)
            + ". Both are comparisons against this video only.")


# The frame every expression finding is stated inside. Not a disclaimer bolted
# on the end — it is the accurate description of what the numbers are, and
# without it a reader takes a label distribution for a record of an experience.
EXPRESSION_FRAME = (
    "All of the above describes labels a five-class classifier assigned to "
    "faces it could see — not what anyone felt. It has no notion of intensity, "
    "degrades on profile, occlusion and motion, and cannot distinguish a "
    "performed expression from a felt one. Treat it as a map of where to look, "
    "not as a finding about a person."
)


def _ratio_phrase(negative: float, positive: float) -> str:
    """"Four to one" and friends, or nothing when one side is empty."""
    if positive <= 0 or negative <= 0:
        return ""
    high, low = max(negative, positive), min(negative, positive)
    factor = high / low
    if factor < 1.5:
        return "about evenly split"
    side = "negative" if negative > positive else "positive"
    return f"{factor:.1f} to 1 {side}"


def summarise_expression_arc(analysis: Mapping) -> list:
    """The video-level reading of the expression scan, as a few plain lines.

    Ordered the way the numbers should be read rather than the way they were
    computed: how much of the file was legible at all, then what it contained,
    then whether it changed, then every reason to doubt it — and the frame last,
    where it is the thing left in the reader's head.
    """
    if not analysis:
        return []

    coverage = analysis.get("coverage") or {}
    valence = analysis.get("valence") or {}
    labels = analysis.get("labels") or {}
    lines = []

    read_pct = float(coverage.get("pct") or 0.0)
    if labels:
        mix = ", ".join(
            f"{name} {stats['share_pct']:.0f}%"
            for name, stats in sorted(labels.items(),
                                      key=lambda kv: -kv[1]["share_pct"])
        )
        lines.append(
            f"A face was readable in {read_pct:.0f}% of the video "
            f"({coverage.get('read_seconds', 0)}s). Across those seconds the "
            f"classifier reported {mix}."
        )

    negative = float(valence.get("negative_pct") or 0.0)
    positive = float(valence.get("positive_pct") or 0.0)
    if negative or positive:
        ratio = _ratio_phrase(negative, positive)
        line = (f"{negative:.0f}% of the read seconds carry a negative-valence "
                f"label and {positive:.0f}% a positive one")
        if ratio:
            line += f" — {ratio}"
        unvalenced = float(valence.get("unvalenced_pct") or 0.0)
        if unvalenced:
            line += (f". A further {unvalenced:.0f}% read as surprise, which "
                     f"carries no direction and is counted on neither side")
        lines.append(line + ".")

    # A split first: it is the more legible structure when a file has one, and
    # the one that answers "did it start one way and change".
    shift = analysis.get("shift") or {}
    if shift:
        lines.append(
            f"The reading changes most at {_clock(shift['at'])}: it averages "
            f"{shift['before']:+.2f} before that point and {shift['after']:+.2f} "
            f"after it ({shift['direction']})."
        )

    arc = analysis.get("arc") or {}
    if arc.get("confident"):
        lines.append(
            f"Across the whole file the drift runs {arc['direction']}, from "
            f"{arc['start_valence']:+.2f} to {arc['end_valence']:+.2f}."
        )
    elif arc.get("direction") in ("toward positive", "toward negative"):
        lines.append(
            f"A straight line through the file would slope {arc['direction']}, "
            f"but it explains only {float(arc.get('fit') or 0.0) * 100:.0f}% of "
            f"the variation — the reading moves in stretches, not as a trend, so "
            f"the split above describes it better than the slope does."
        )
    elif arc.get("direction") == "flat":
        lines.append("The reading does not trend in either direction across the "
                     "file — what structure it has is local.")

    episodes = [e for e in (analysis.get("episodes") or []) if e["sign"] < 0][:3]
    if episodes:
        where = ", ".join(f"{_clock(e['start'])}–{_clock(e['end'])}"
                          for e in episodes)
        lines.append(f"The longest negative-reading stretches are {where}.")
    positives = [e for e in (analysis.get("episodes") or []) if e["sign"] > 0][:3]
    if positives:
        where = ", ".join(f"{_clock(e['start'])}–{_clock(e['end'])}"
                          for e in positives)
        lines.append(f"The longest positive-reading stretches are {where}.")

    lines.extend(_class_lines(analysis))

    for reason in (analysis.get("reliability") or {}).get("reasons") or ():
        lines.append(f"Caution: {reason}.")

    lines.append(EXPRESSION_FRAME)
    return lines


def _class_lines(analysis: Mapping) -> list:
    """Whether the reading differs while each detected class is on screen.

    The question this answers is the one people arrive with — "is it different
    during X?" — and the honest answer is usually no. Saying so explicitly is
    the whole value: a reader who is told the reading is flat across every class
    stops attributing a file-wide number to one thing in it, and a reader who is
    told nothing goes on assuming the connection they came for.

    A difference, when there is one, is stated as a difference and nothing more.
    The same seconds carry the shot, the lighting and the angle, so which of
    them moved the reading is not something these numbers can separate.
    """
    rows = analysis.get("by_class") or []
    if not rows:
        return []

    baseline = float((analysis.get("valence") or {}).get("mean_all_read") or 0.0)
    differing = [r for r in rows if r.get("distinguishable")]
    if not differing:
        listed = ", ".join(
            f"{r['name']} {float(r['delta']):+.2f} over {r['read_seconds']}s"
            for r in rows[:5]
        )
        return [
            f"No detected class reads differently from the video's own baseline "
            f"of {baseline:+.2f} — {listed}. The reading is a property of this "
            f"file as a whole, not of what is on screen at the time, so it "
            f"cannot be attributed to any one of them."
        ]

    lines = []
    for row in differing[:3]:
        way = "more positive" if float(row["delta"]) > 0 else "more negative"
        lines.append(
            f"While {row['name']} is on screen the reading averages "
            f"{float(row['valence']):+.2f} against the file's {baseline:+.2f} — "
            f"{way}, over {row['read_seconds']} readable seconds, mostly "
            f"{row['dominant']}. That the two differ is measurable; what caused "
            f"the difference is not — the same seconds carry the shot, the "
            f"angle and the lighting too."
        )
    return lines


def _clock(seconds: float) -> str:
    """Local ``M:SS`` so this module needs nothing from the report to render."""
    total = int(float(seconds or 0))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def describe_segment_reading(row: Mapping, video_mean: float = 0.0) -> str:
    """One kept clip's expression reading, against the file's own.

    The delta carries the sentence. In a file that reads negative throughout,
    every clip's absolute figure is negative, and printing it without the
    comparison invites the reader to find meaning in what is merely the video's
    baseline.
    """
    if not row or "valence" not in row:
        return ""
    delta = float(row.get("delta") or 0.0)
    dominant = str(row.get("dominant") or "")
    read = int(row.get("read_seconds") or 0)
    if abs(delta) < 0.15:
        standing = "in line with the rest of the video"
    elif delta > 0:
        standing = "reading more positive than the video's own average"
    else:
        standing = "reading more negative than the video's own average"
    return (f"Expression here is mostly {dominant} over {read} readable second"
            f"{'' if read == 1 else 's'}, {standing} "
            f"({float(row['valence']):+.2f} against {video_mean:+.2f}).")


def describe_all(report: Mapping) -> list:
    """A sentence per clip, each ranked against the others in the cut."""
    segments = list(report.get("segments") or [])
    peers = [float(s.get("score") or 0.0) for s in segments]
    return [describe(s, peers) for s in segments]


def summarise_run(report: Mapping) -> str:
    """One sentence about the highlight as a whole.

    Not "how many clips were exceptional" — every kept clip outscored the rest
    of the video, that is what being kept means. What is worth saying is
    whether the scores separated at all, and how much of the video the cut
    actually saw.
    """
    segments = list(report.get("segments") or [])
    if not segments:
        return "Nothing was selected."

    scores = [float(s.get("score") or 0.0) for s in segments]
    duration = float((report.get("video") or {}).get("duration") or 0.0)
    span = (max(s["end"] for s in segments) - min(s["start"] for s in segments)
            if duration else 0.0)

    if len(segments) > 2 and max(scores) == min(scores):
        return (f"All {len(segments)} clips scored identically, so nothing "
                "separated them — the order they were picked in was arbitrary.")

    parts = [f"{len(segments)} clips"]
    if max(scores) > 0:
        spread = (max(scores) - min(scores)) / max(scores) * 100.0
        parts.append(f"the best scoring {spread:.0f}% above the weakest"
                     if spread >= 5 else "all scoring within a few points")
    if duration and span:
        parts.append(f"drawn from {span / duration * 100:.0f}% of its length")
    return _join(parts).capitalize() + "."
