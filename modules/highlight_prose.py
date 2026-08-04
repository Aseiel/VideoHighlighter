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


def _as_sentences(parts: Sequence[str], lead: str = "") -> str:
    """Where the clip stands, then what it was chosen on. Two sentences, not one.

    Four measurements chained with commas reads as one long breath and gets
    skipped. Splitting them changes nothing about what is claimed and a great
    deal about whether anyone finishes the line.

    The break falls after the ranking because that is the natural seam: where
    this clip stands is one thought, what fired is another. It deliberately does
    *not* split further and capitalise each clause, which was tried and was
    worse — half these clauses begin with detected names ("jumping recognised
    at 0.88"), and sentence-casing them rewrites data the detector produced.
    A fixed lead-in carries the second sentence instead, so nothing measured
    ever has to start one.
    """
    parts = [p for p in parts if p]
    if not parts:
        return f"{lead}." if lead else ""
    head, rest = parts[0], parts[1:]
    if lead:
        first = f"{lead} — {head}."
    else:
        first = f"{head[0].upper()}{head[1:]}."
    if not rest:
        return first
    return f"{first} Chosen on {_join(rest)}."


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
        return _as_sentences(parts, lead=standing)
    if not parts:
        return ""
    return _as_sentences(parts[1:], lead=parts[0][0].upper() + parts[0][1:])


# A size claim resting on a box the detector was this unsure of should carry the
# number that undermines it. Same threshold the expression classifier uses for
# "I am picking between near-ties".
WEAK_DETECTION = 0.5


def _subject_line(subject: Mapping) -> tuple:
    """What is unusual about one detected class here, as ``(kind, line)``.

    The kind is returned so the caller can keep the paragraph varied — three
    findings of the same shape read as one finding repeated. See
    :func:`explain_standout`.

    The order of preference is the order of trustworthiness. A ratio against
    something else in the same frame comes first because moving the camera
    scales both boxes and leaves it unchanged; bare frame share comes second and
    carries the caveat that it cannot separate a larger subject from a closer
    camera; rarity comes last and needs no size claim at all.
    """
    name = str(subject.get("name") or "").strip()
    if not name:
        return "", ""

    relative = subject.get("relative") or {}
    lead = ""
    kind = ""
    median = float(relative.get("median") or 0.0)
    ratio = float(relative.get("ratio") or 0.0)
    # Ranked high *and* actually different. Either alone is a sentence that
    # misleads: a big lead over nothing, or a big number that is the norm here.
    big_enough = (median > 0 and abs(ratio / median - 1.0) >= MIN_RATIO_EFFECT)
    if (relative.get("enough_samples") and big_enough
            and float(relative.get("percentile") or 0.0) >= NOTABLE):
        linear = float(relative.get("linear_ratio")
                       or math.sqrt(max(0.0, float(relative["ratio"]))))
        kind = "relative"
        lead = (
            f"{name} covers {float(relative['ratio']):.1f}× the area of the "
            f"{relative['reference']} beside it — about {linear:.1f}× across — "
            f"larger than in {float(relative['percentile']):.0f}% of this "
            f"video's {relative['stretch_seconds']}s stretches where the two "
            f"share a frame, against a usual {float(relative['median']):.1f}×"
        )
    elif (subject.get("enough_samples")
          and float(subject.get("frame_share_percentile") or 0.0) >= NOTABLE):
        kind = "frame_share"
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
        kind = kind or "rarity"
        rarity = (f"{name} is in only {float(prevalence):.0f}% of the video's "
                  f"detected seconds")

    if lead and rarity:
        # The rarity clause has already named the class; don't say it twice.
        lead = f"{lead}, and it is a class in only {float(prevalence):.0f}% of " \
               f"the video's detected seconds"
    elif rarity:
        lead = rarity
    if not lead:
        return "", ""

    caveats = []
    presence = subject.get("clip_presence_pct")
    if presence is not None and float(presence) < FLEETING_PRESENCE:
        caveats.append(f"present for only {float(presence):.0f}% of the clip")
    confidence = subject.get("confidence")
    if confidence is not None and float(confidence) < WEAK_DETECTION:
        caveats.append(f"on a {float(confidence):.2f} detection")
    if caveats:
        lead = f"{lead} — {_join(caveats)}"
    return kind, lead[0].upper() + lead[1:] + "."


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


# At most this many size/rarity findings under one clip, and at most this many
# of any single kind. Measured on a real run: three detected classes produced
# three frame-share sentences in a row, which reads as one sentence written
# three times and pushes the loudness and expression readings out of sight.
MAX_SUBJECT_LINES = 2
MAX_LINES_PER_KIND = {"relative": 1, "frame_share": 1, "rarity": 2}


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
    lines: list = []
    used: dict = {}
    for subject in comparison.get("subjects") or ():
        kind, line = _subject_line(subject)
        if not line:
            continue
        # Three findings of the same shape read as one finding repeated, and a
        # paragraph of them buries whatever else the clip had to say. The frame
        # -share reading is the one that multiplies -- every class in a frame has
        # a share, so an unconstrained list becomes a table of areas -- and it is
        # also the weakest, since it cannot tell a larger subject from a closer
        # camera. One of those is plenty.
        if used.get(kind, 0) >= MAX_LINES_PER_KIND.get(kind, 1):
            continue
        used[kind] = used.get(kind, 0) + 1
        lines.append(line)
        if len(lines) >= MAX_SUBJECT_LINES:
            break
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


def _lift_phrase(lift: float) -> str:
    """Turn a ratio into words, in the direction it actually points.

    "0.4x as much" is arithmetic; "less than half as much" is the same fact in
    a form a reader takes in at a glance. Below parity the ratio is inverted so
    the sentence never asks anyone to reason about a fraction.
    """
    if lift <= 0:
        return ""
    if lift >= 1.0:
        return f"{lift:.1f}× as much" if lift >= 1.15 else "about as much"
    return f"{1.0 / lift:.1f}× less"


# How close a chapter's best second has to come to the cut threshold before the
# miss is called narrow. Within a tenth is close enough that a small weight
# change would flip it; further out and saying "nearly" would be flattery.
NEAR_MISS_SHARE = 0.9


def _why_not_selected(chapter: Mapping) -> list:
    """Why nothing was taken from this stretch — the three reasons differ.

    "Nothing was selected" is true of most chapters of most videos and tells a
    reader nothing they can act on. The useful question is which of three things
    happened, because each has a different fix:

    *Nothing fired here.* No detector scored a single second. Raising weights
    cannot help; the stretch is invisible to whatever was run, and the fix is a
    detector, not a number.

    *It scored, but under the bar.* The gap to the weakest clip that was kept is
    the actionable figure, and the signals that did fire name which weight to
    raise.

    *It scored well enough and still lost.* The cut filled before reaching it —
    a length or coverage limit, not a scoring problem. Telling someone to raise
    a weight here would waste their afternoon.
    """
    lines = ["Nothing from this chapter was selected."]

    peak = chapter.get("score_peak")
    threshold = chapter.get("cut_threshold")
    fired = list(chapter.get("signals_present") or [])

    if peak is None:
        return lines

    peak = float(peak)
    if peak <= 0:
        lines.append("Nothing here scored at all — no detector fired in this "
                     "stretch, so no weighting would have pulled a clip out of "
                     "it. It is invisible to what was run, not merely weaker.")
        return lines

    stamp = chapter.get("score_peak_second")
    where = f" at {_clock(float(stamp))}" if stamp is not None else ""

    if threshold is None:
        lines.append(f"Its best second scored {peak:.0f}{where}.")
        return lines

    threshold = float(threshold)
    if peak >= threshold:
        lines.append(
            f"Its best second scored {peak:.0f}{where}, which clears the "
            f"{threshold:.0f} the weakest kept clip managed — so this stretch "
            f"lost to length rather than to score. The cut filled up before it. "
            f"Raising a weight will not change that; a longer highlight, or "
            f"more even coverage, would.")
        return lines

    gap = threshold - peak
    close = peak >= threshold * NEAR_MISS_SHARE
    lead = ("Came close: its" if close else "Its")
    lines.append(
        f"{lead} best second scored {peak:.0f}{where}, against {threshold:.0f} "
        f"for the weakest clip that made the cut — short by {gap:.0f}.")
    if fired:
        # The reader tunes weights by the names on the settings table, so the
        # sentence has to use those and not the internal keys.
        try:
            from modules.highlight_report import SIGNAL_LABELS
            labels = dict(SIGNAL_LABELS)
        except Exception:
            labels = {}
        named = [labels.get(key, key) for key in fired]
        lines.append(
            f"What did fire here: {_join(named)}. Raising one of those weights "
            f"is what would bring this stretch in.")
    else:
        lines.append("No signal scored a point here, so there is no weight to "
                     "raise — this stretch needs a detector it does not have.")
    return lines


def describe_chapter(chapter: Mapping) -> list:
    """What marks one chapter out from the rest of its video.

    Deliberately willing to return a single flat sentence. Most chapters of most
    videos are not distinctive, and a breakdown that finds a story in every one
    of them is telling the reader nothing they can act on — the chapters that
    *are* different only mean something against neighbours that are not.
    """
    from modules.chapter_compare import CUT_SHARE_LIFT, distinctive

    lines = []

    # Where the cut came from. First because it needs no detector, so it is the
    # one line that survives on a video nothing was run against.
    clips = int(chapter.get("clips") or 0)
    share_lift = float(chapter.get("cut_share_lift") or 0.0)
    runtime_share = float(chapter.get("runtime_share_pct") or 0.0)
    cut_share = float(chapter.get("cut_share_pct") or 0.0)
    if clips and share_lift >= CUT_SHARE_LIFT:
        lines.append(
            f"Supplied {cut_share:.0f}% of the cut from {runtime_share:.0f}% of "
            f"the runtime — {share_lift:.1f}× its share, across {clips} clip"
            f"{'' if clips == 1 else 's'}.")
    elif not clips:
        lines.extend(_why_not_selected(chapter))

    # What this stretch is mostly made of, and what changed at its start. These
    # come before the video-wide comparisons because they are what turns a list
    # of chapters into a sequence: "what is this" then "what changed", rather
    # than eleven independent verdicts against the same average.
    dominant = chapter.get("dominant") or {}
    if dominant:
        lines.append(f"Mostly {dominant['name']} — on screen for "
                     f"{float(dominant['share_pct']):.0f}% of this chapter's "
                     f"detected seconds.")

    for change in (chapter.get("changes") or []):
        was, now = float(change["from_pct"]), float(change["to_pct"])
        if change["direction"] == "rose":
            lines.append(f"{change['name']} takes over here — up from "
                         f"{was:.0f}% of the previous chapter to {now:.0f}%.")
        else:
            lines.append(f"{change['name']} drops back — {was:.0f}% of the "
                         f"previous chapter, {now:.0f}% of this one.")

    for finding in distinctive(chapter):
        phrase = _lift_phrase(float(finding["lift"]))
        if not phrase:
            continue
        seconds = int(finding.get("seconds") or 0)
        # "1.4x less" takes "than"; "1.4x as much" takes "as". A single fixed
        # joiner produced "1.4x less as across the video" on every chapter where
        # a class was under-represented, which is most of them.
        joiner = "as" if phrase.endswith("as much") else "than"
        if finding["kind"] == "expression":
            lines.append(
                f"Reads {finding['name']} for {finding['chapter_share_pct']:.0f}% "
                f"of its readable seconds — {phrase} {joiner} the video overall "
                f"({finding['video_share_pct']:.0f}%), over {seconds}s.")
        else:
            lines.append(
                f"{finding['name']} is on screen for "
                f"{finding['chapter_share_pct']:.0f}% of this chapter's detected "
                f"seconds — {phrase} {joiner} across the video "
                f"({finding['video_share_pct']:.0f}%), over {seconds}s.")

    # Pace and level describe how a stretch was shot and mixed rather than what
    # is in it, so they come last and only when they are clearly off the video's
    # own norm.
    pace_lift = float(chapter.get("pace_lift") or 0.0)
    if pace_lift and (pace_lift >= 1.5 or pace_lift <= 0.67):
        direction = "faster" if pace_lift > 1 else "slower"
        lines.append(f"Cuts {direction} than the video's average — "
                     f"{float(chapter.get('shots_per_minute') or 0):.0f} shots a "
                     f"minute, {_lift_phrase(pace_lift)}.")

    level = float(chapter.get("loudness_delta_db") or 0.0)
    if abs(level) >= 3.0:
        lines.append(f"Runs {abs(level):.0f} dB "
                     f"{'louder' if level > 0 else 'quieter'} than the video's "
                     "median level.")

    if not lines:
        lines.append("Nothing here separates it from the rest of the video.")
    return lines


def summarise_chapter_run(chapters: Sequence[Mapping]) -> str:
    """One sentence about the chapter breakdown as a whole.

    The useful fact is concentration — whether the highlights came from
    everywhere or from one stretch. A cut drawn entirely from two chapters of
    twelve is a finding about the video; the same clips spread evenly is a
    finding about the detector.
    """
    chapters = list(chapters or [])
    if not chapters:
        return ""
    if len(chapters) == 1:
        return ("The video was not divided — nothing in its shot structure "
                "separated one stretch from another.")

    with_clips = [c for c in chapters if int(c.get("clips") or 0)]
    method = str(chapters[0].get("method") or "")
    basis = ("its own shot lengths" if method == "shot-length"
             else "where the footage stops looking like what came before")

    if not with_clips:
        return (f"{len(chapters)} chapters, divided on {basis}. "
                "No clips were selected from any of them.")

    share = 100.0 * len(with_clips) / len(chapters)
    lead = max(chapters, key=lambda c: float(c.get("cut_share_pct") or 0.0))
    tail = (f" The largest single contribution is {lead['title']}, at "
            f"{float(lead.get('cut_share_pct') or 0):.0f}% of the cut."
            if float(lead.get("cut_share_pct") or 0) > 0 else "")
    return (f"{len(chapters)} chapters, divided on {basis}. Clips came from "
            f"{len(with_clips)} of them ({share:.0f}%).{tail}")


# How far above the video's own middle a moment has to sit before the sentence
# reaches for a strong word. Calibrated on measured material: confirmed
# stand-out moments landed between +7 and +24 dB against the median, so the top
# band has to start high enough that most of them do not qualify for it.
FAR_ABOVE_DB = 18.0
WELL_ABOVE_DB = 10.0
ABOVE_DB = 4.0


def _level_phrase(vs_video_db: Optional[float]) -> str:
    """How the peak sat against the video, in words rather than decibels."""
    if vs_video_db is None:
        return ""
    if vs_video_db >= FAR_ABOVE_DB:
        return "far above"
    if vs_video_db >= WELL_ABOVE_DB:
        return "well above"
    if vs_video_db >= ABOVE_DB:
        return "above"
    if vs_video_db <= -ABOVE_DB:
        return "below"
    return "close to"


def describe_loudest(entry: Mapping) -> str:
    """Where a clip was loudest, and what was on screen, in one sentence.

    Says *loud*, never *why* it was loud. The distinction is not pedantry: on
    measured material the same acoustic signature appeared during three
    different kinds of moment, and a sentence that named the kind would have
    been wrong two times in three while reading exactly as confidently.
    """
    loud = entry.get("loudest") or {}
    if not loud:
        return ""
    stamp = str(loud.get("timestamp", ""))
    vs = loud.get("vs_video_db")
    phrase = _level_phrase(vs if vs is None else float(vs))

    names = [str(c) for c in (loud.get("classes") or [])]
    where = _join(names)
    if where:
        verb = "was" if len(names) == 1 else "were"
        tail = f" {where} {verb} on screen at that second."
    else:
        tail = " Nothing was labelled at that second."

    if phrase in ("far above", "well above"):
        return (f"Loudest at {stamp}, {phrase} this video's usual level "
                f"({float(vs):+.0f} dB).{tail}")
    if phrase == "above":
        return (f"Loudest at {stamp}, a little above the video's usual level "
                f"({float(vs):+.0f} dB).{tail}")
    if phrase == "below":
        return (f"Loudest at {stamp}, but still quieter than the video "
                f"typically is ({float(vs):+.0f} dB).{tail}")
    return (f"Loudest at {stamp}, about as loud as the video usually "
            f"is.{tail}")


def describe_motion_peak(entry: Mapping) -> str:
    """Where movement spiked and then stopped inside this clip.

    Named for the shape the detector actually looks for -- a burst above the
    scene's own average, followed by several seconds below it -- rather than
    for "action", which is what the number gets read as. The distinction earns
    its keep: continuous movement never produces the stillness the rule needs,
    so on footage that does not pause, these mark where activity *ended*
    (often a cut) and not where it was most intense.
    """
    peak = entry.get("motion_peak") or {}
    if not peak:
        return ""
    count = int(peak.get("count") or 1)
    stamp = str(peak.get("timestamp", ""))
    if count > 1:
        return (f"Movement spiked at {stamp} and dropped away after — the "
                f"burst-then-stillness this signal scores. {count} of them fall "
                f"inside this clip.")
    return (f"Movement spiked at {stamp} and dropped away after — the "
            f"burst-then-stillness this signal scores.")


def summarise_level_by_class(data: Mapping) -> list:
    """The per-class level comparison, said plainly — including when it says nothing.

    Three sentences at most, and the third is the one that matters: loudness
    measures emphasis, not cause. Stated once here rather than hedged into every
    other sentence, so the prose stays readable while the limit stays visible.
    """
    data = data or {}
    rows = list(data.get("classes") or [])
    if not rows:
        return []

    lines = []

    loudest = data.get("loudest") or {}
    if loudest:
        where = _join([str(c) for c in (loudest.get("classes") or [])])
        stamp = str(loudest.get("timestamp", ""))
        if where:
            lines.append(f"The loudest labelled second in the whole video is "
                         f"{stamp}, with {where} on screen.")
        else:
            lines.append(f"The loudest labelled second in the whole video is "
                         f"{stamp}.")

    comp = data.get("comparison") or {}
    if comp:
        louder, quieter = comp.get("louder"), comp.get("quieter")
        diff = abs(float(comp.get("median_difference_db") or 0.0))
        pairs = int(comp.get("pairs") or 0)
        if comp.get("resolvable"):
            lines.append(
                f"Across {pairs} separate stretches, this video was "
                f"consistently louder during {louder} than during {quieter} — "
                f"by about {diff:.1f} dB. Because each stretch was compared "
                f"with nearby material rather than with the video as a whole, "
                f"that difference is not just a matter of where in the video "
                f"each happened.")
        else:
            mde = float(comp.get("min_detectable_db") or 0.0)
            lines.append(
                f"{louder} measured about {diff:.1f} dB louder than {quieter}, "
                f"but that is inside the margin: across {pairs} stretches the "
                f"readings scatter enough that nothing smaller than "
                f"{mde:.1f} dB can be told from noise here. Treat the two as "
                f"indistinguishable in this video — which is not the same as "
                f"saying they would be in another.")
    elif len(rows) > 1:
        lines.append("The classes never occurred close enough together in time "
                     "to be compared fairly, so no difference is claimed.")

    # The limit, once. Everything above is about level; none of it is about
    # cause, and on measured material the same signature covered three
    # different kinds of moment.
    lines.append(
        "All of this measures how loud, not why. The same acoustic signature "
        "turns up on different kinds of moment, so these numbers say where the "
        "emphasis fell — what it meant is a judgement for whoever watches it.")
    return lines


# Two marked seconds this close are the same event seen twice, not one following
# another. Below it "came before" is a claim about detector timing, not content.
TOGETHER_SECONDS = 2

# A gap wider than this is two things that happened in the same clip, not a
# sequence -- saying one followed the other would invent a link across half a
# minute of unexamined footage.
SEQUENCE_SECONDS = 20


def describe_signal_relations(entry: Mapping) -> str:
    """How the seconds this clip named relate to each other.

    The report marks a loudest second and a motion peak and, until now, left
    them as two unconnected facts. Their *order* is the thing a reader
    reconstructs by hand otherwise, and it is free: both are already measured.

    Says which came first and how far apart, and nothing about why. One clip
    cannot distinguish "the movement stopped and then she reacted" from two
    unrelated events sharing thirty seconds, and the sentence must not pretend
    otherwise -- :func:`summarise_signal_relations` is where a repeated ordering
    becomes evidence, because that needs the whole run.
    """
    loud = entry.get("loudest") or {}
    motion = entry.get("motion_peak") or {}
    if loud.get("second") is None or motion.get("second") is None:
        return ""
    gap = int(loud["second"]) - int(motion["second"])
    if abs(gap) <= TOGETHER_SECONDS:
        return (f"Both landed together — movement stopped and the loudest point "
                f"arrived within {abs(gap)}s of each other.")
    if abs(gap) > SEQUENCE_SECONDS:
        return (f"The two are {abs(gap)}s apart in this clip, far enough that "
                f"they are separate events rather than one following the other.")
    if gap > 0:
        return (f"Movement stopped first, at {motion['timestamp']}; the loudest "
                f"point came {gap}s later, at {loud['timestamp']}.")
    return (f"The loudest point came first, at {loud['timestamp']}; movement "
            f"dropped away {abs(gap)}s later, at {motion['timestamp']}.")


# How many clips must carry both marks before their ordering is worth a claim.
MIN_CLIPS_FOR_PATTERN = 4

# And how consistently they must agree. Two-thirds is the point at which the
# ordering is describing the footage rather than the four clips that happened
# to land that way.
PATTERN_AGREEMENT = 0.67


def summarise_signal_relations(segments: Sequence[Mapping]) -> str:
    """Whether the ordering repeats across the run, which is what makes it a finding.

    One clip where movement stopped nine seconds before the loudest point is a
    coincidence. Seven of nine is a property of the footage, and it is the kind
    of thing a person would otherwise only notice after watching the whole
    thing twice. Returns nothing when the clips disagree -- a split verdict is
    the honest output of a split measurement.
    """
    gaps = []
    for e in segments or ():
        loud = (e.get("loudest") or {}).get("second")
        motion = (e.get("motion_peak") or {}).get("second")
        if loud is None or motion is None:
            continue
        gap = int(loud) - int(motion)
        if abs(gap) <= SEQUENCE_SECONDS:
            gaps.append(gap)
    if len(gaps) < MIN_CLIPS_FOR_PATTERN:
        return ""

    after = [g for g in gaps if g > TOGETHER_SECONDS]
    before = [g for g in gaps if g < -TOGETHER_SECONDS]
    together = [g for g in gaps if abs(g) <= TOGETHER_SECONDS]
    total = len(gaps)

    if len(after) / total >= PATTERN_AGREEMENT:
        median = sorted(abs(g) for g in after)[len(after) // 2]
        return (f"In {len(after)} of {total} clips the loudest point arrives "
                f"after movement has stopped, typically about {median}s later. "
                f"That ordering is a property of this footage, not of any one "
                f"clip — what it means is still a matter for whoever watches it.")
    if len(before) / total >= PATTERN_AGREEMENT:
        median = sorted(abs(g) for g in before)[len(before) // 2]
        return (f"In {len(before)} of {total} clips the loudest point comes "
                f"first and movement drops away about {median}s afterwards.")
    if len(together) / total >= PATTERN_AGREEMENT:
        return (f"In {len(together)} of {total} clips the loudest point and the "
                f"motion peak land within {TOGETHER_SECONDS}s of each other.")
    return (f"Across {total} clips the two land in no consistent order — "
            f"{len(after)} with sound after movement, {len(before)} before, "
            f"{len(together)} together. Nothing here links them.")
