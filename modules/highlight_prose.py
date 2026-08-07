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
    """The deeper reading of one clip, subjects and expression in one list.

    Kept as the flat form because callers and tests read it that way;
    :func:`clip_sections` is what files each line under the signal that produced
    it.
    """
    found = _standouts(entry)
    return found["subjects"] + found["expression"]


def _standouts(entry: Mapping) -> dict:
    """The deeper reading of one clip, filed by the signal that produced it.

    Separate from :func:`describe` on purpose. That sentence is a summary and
    has to stay one line; this is the evidence a reader turns to when they want
    to argue with the pick, and cramming it into the same sentence would cost
    both of them their job.

    Returns empty lists when nothing cleared the thresholds — which is the
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
    return {"subjects": lines, "expression": [expression] if expression else []}


def clip_sections(entry: Mapping,
                  reading: Optional[Mapping] = None,
                  video_valence: float = 0.0,
                  peer_scores: Optional[Sequence[float]] = None) -> list:
    """One clip's evidence, grouped under the signal that produced it.

    The same shape as :func:`conclude`, for the same reason. A clip's findings
    arrived as six sentences in a column, each true and each starting from a
    different measurement, and a reader who wanted to know what the *sound* did
    had to identify which of the six was about sound before they could read it.
    Under a heading that question is answered before the line is read — which is
    also what makes a connection between two signals visible, since they are now
    two named things rather than six sentences.

    The lead sentence stays outside this: it is the one line that is about the
    pick rather than about any one signal, and filing it under a heading would
    make it look like evidence for one of them.

    Returns ``[(heading, [line, ...]), ...]``, headings only where a measurement
    exists to fill them.
    """
    found = _standouts(entry)
    sections = []

    motion = describe_motion_peak(entry) if _motion_scored(entry) else ""
    if motion:
        sections.append(("Movement", [motion]))

    loud = describe_loudest(entry)
    if loud:
        sections.append(("Sound", [loud]))

    face = [line for line in (describe_expression_peak(entry),
                              describe_reading_shot(entry),
                              *found["expression"],
                              describe_segment_reading(reading or {},
                                                       video_valence))
            if line]
    if face:
        sections.append(("Face expression", face))

    on_screen = list(found["subjects"])
    arrival = describe_arrival_shot(entry)
    if arrival:
        on_screen.append(arrival)
    if on_screen:
        sections.append(("On screen", on_screen))

    # The ordering last and on its own, because it is the only part that is
    # about more than one signal — under any single heading it would read as
    # that signal's finding rather than as the relation between them.
    summary = summarise_clip(entry, peer_scores)
    if summary:
        sections.append(("Summary", summary))
    return sections


def summarise_clip(entry: Mapping,
                   peer_scores: Optional[Sequence[float]] = None) -> list:
    """The clip in plain words, with no figure in it.

    The sections above are the evidence and are full of numbers, because that
    is what evidence is. This is the part a reader consults *instead of* reading
    them, and a summary that says "movement drops away 12s later and the loudest
    point arrives 15s after that" has handed the work straight back: the reader
    still has to decide whether twelve seconds is a lot.

    So the same facts arrive as description. Where it stood among the clips
    kept, what happened in what order, and whether this clip is busier or
    louder than the video it came from — each one already measured, each one
    said the way a person would say it.

    It reads as a paragraph rather than as four findings stacked up. Each of
    these was its own bullet once, and a clip described in four clauses that
    never refer to each other reads like a form someone filled in — which is
    what a reader is being spared here, not given a compressed version of.

    "In order" survives the rewrite and is doing the same work the longer phrase
    did. Prose about one moment reads as a story, and a story reads as a cause;
    two words at the front keep the sentence an ordering without narrating the
    caution.
    """
    said = []

    opening = _standing_and_evidence(entry, peer_scores)
    if opening:
        said.append(opening)

    # The order and the clip's character in one sentence: they are the same
    # thought — what happened, and what it was like — and splitting them was
    # what made the summary read as a list of separate verdicts.
    chain = _clip_chain(entry)
    character = _clip_character(entry)
    if chain and character:
        said.append(f"{chain[:-1]} — {character[0].lower()}{character[1:]}")
    elif chain:
        said.append(chain)
    elif character:
        said.append(character)

    unusual = describe_combination(entry)
    if unusual:
        said.append(unusual)
    return [" ".join(said)] if said else []


# How much of a video has to carry a combination before it stops being unusual.
# Deliberately strict at the top: four signals agreeing looks like a finding, and
# the only thing that makes it one is the video not doing it constantly.
RARE_COMBINATION = 10.0
UNCOMMON_COMBINATION = 25.0
COMMON_COMBINATION = 50.0

# Below this many marks there is no combination to speak of -- one mark is a
# measurement, and "this video has loud seconds elsewhere" is not a finding.
MIN_COMBINATION_MARKS = 2


def describe_combination(entry: Mapping) -> str:
    """Whether this clip's combination of marks is unusual for this video.

    The question a reader reaches for after the ordering, and the one the report
    could not previously answer: four signals agreeing looks like a finding, and
    whether it *is* one depends entirely on how often this video does that
    anyway. On footage where every stretch carries the same four marks, a clip
    carrying them is the norm wearing the costume of a discovery.

    This is also the only claim in the report shaped like a probability, and the
    shape is exact: a frequency over comparable stretches of this one file. It
    says how often the combination occurs, not what it means — a rare
    combination is a good place to look, and no arrangement of these marks
    supports a reading of what happened in it.
    """
    combination = entry.get("combination") or {}
    marks = list(combination.get("marks") or ())
    if len(marks) < MIN_COMBINATION_MARKS or not combination.get("windows"):
        return ""

    from modules.signal_combinations import MARK_NAMES
    named = _join([MARK_NAMES.get(m, m) for m in marks])
    pct = float(combination.get("pct") or 0.0)

    if pct <= RARE_COMBINATION:
        return ("Hardly anywhere else in this video do those land together, "
                "which is what makes the clip worth a look.")
    if pct <= UNCOMMON_COMBINATION:
        return "Not many other stretches of this video put those together."
    if pct <= COMMON_COMBINATION:
        return ("A fair part of the video does the same thing, so this is not "
                "the only place to look.")
    return ("Most of the video does the same thing, though — so the signals "
            "agreeing here says little about this clip in particular.")


def _standing_and_evidence(entry: Mapping,
                           peer_scores: Optional[Sequence[float]]) -> str:
    """Where it stood and what fired, in one sentence and no figures.

    The same two facts :func:`describe` opens with, minus the percentile, the
    detection confidence and the dBFS — every one of which is already printed
    under "show the measurements" a few lines below. Saying them twice was how
    the summary came to be the most number-dense line on the card.
    """
    standing = _standing(entry, peer_scores)
    evidence = _evidence(entry)
    if not evidence:
        return f"{standing}." if standing else ""

    only = "" if len(entry.get("signals_present") or []) != 1 else " alone"
    agreement = _agreement_words(entry)
    chosen = f"chosen on {evidence}{only}"
    if agreement:
        chosen += f", {agreement}"
    if standing:
        return f"{standing}, {chosen}."
    return f"{chosen[0].upper()}{chosen[1:]}."


def _agreement_words(entry: Mapping) -> str:
    """:func:`_agreement`, with the spread said rather than counted."""
    measured = entry.get("measured") or {}
    if len(entry.get("signals_present") or []) < 2:
        return ""
    if not measured.get("signals_coincide"):
        return "though not at the same instant"
    spread = float(measured.get("signal_spread_seconds") or 0.0)
    if spread <= 0:
        return "all landing on the same second"
    if spread <= TOGETHER_SECONDS:
        return "landing within a moment of each other"
    if spread <= 5:
        return "landing within a few seconds of each other"
    return "landing several seconds apart"


def _clip_chain(entry: Mapping) -> str:
    """The clip's marked seconds as an order, with the gaps in words."""
    marks = _longest_chain(_sequence_marks(entry))
    carries_arrival = any(m[3] == "event" for m in marks)
    if len(marks) < (2 if carries_arrival else 3):
        return ""

    clauses = [marks[0][1]]
    for i, (second, clause, _stamp, _kind) in enumerate(marks[1:], start=1):
        gap = second - marks[i - 1][0]
        if gap <= TOGETHER_SECONDS:
            clauses.append(f"{clause} in the same moment")
        elif i == 1:
            clauses.append(f"{clause} {_gap_words(gap)}")
        else:
            words = _gap_words(gap).replace(" later", " after that")
            clauses.append(f"{clause} {words}")
    return f"In order: {_join(clauses)}."


def _clip_character(entry: Mapping) -> str:
    """Whether this clip is busier or louder than the video it came from.

    The two questions a reader actually has about a moment they cannot see --
    was there more going on here than usual, was it louder than usual -- and
    both are already measured against this video's own distribution. Said in
    words because a percentile is the reader doing the comparison themselves.
    """
    parts = []

    vs = (entry.get("loudest") or {}).get("vs_video_db")
    if vs is not None:
        phrase = _level_phrase(float(vs))
        if phrase in ("far above", "well above"):
            parts.append("much louder than this video usually runs")
        elif phrase == "above":
            parts.append("a little louder than this video usually runs")
        elif phrase == "below":
            parts.append("quieter than this video usually runs")

    # The busiest of the movement signals, whichever the run happened to score.
    signals = (entry.get("measured") or {}).get("signals") or {}
    ranks = [float(signals[key].get("percentile") or 0.0)
             for key in ("motion_event", "motion_peak")
             if key in signals and signals[key].get("percentile") is not None]
    if ranks:
        rank = max(ranks)
        if rank >= EXCEPTIONAL:
            parts.append("with more movement in it than almost anywhere else "
                         "in the video")
        elif rank >= NOTABLE:
            parts.append("with more movement than this video usually carries")
        elif rank <= ORDINARY:
            parts.append("with less movement than this video usually carries")

    if not parts:
        return ""
    return _join(parts).capitalize() + "."


def _motion_scored(entry: Mapping) -> bool:
    """Whether the motion peak actually earned points here.

    A signal that fired but scored nothing is in the breakdown already, and
    narrating it under its own heading would imply it drove the pick.
    """
    return float((entry.get("breakdown") or {}).get("motion_peak") or 0.0) > 0


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

    # What was said, when a transcript was run. Placed with the other "what is
    # this stretch" lines rather than with the video-wide comparisons below,
    # because on footage where the detectors have little to distinguish it is
    # the only line that says what the chapter is *about* rather than how it
    # ranks. Empty when no transcript reached the report.
    lines.extend(describe_speech(chapter))

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


def describe_speech(chapter: Mapping) -> list:
    """What the soundtrack of one chapter was, in sentences. ``[]`` if silent.

    Every clause is a threshold over a figure :mod:`modules.chapter_speech`
    measured, exactly like the rest of this module — the transcript widens what
    can be described, it does not license a guess. In particular the distinctive
    words are introduced as *the words this stretch used and the others did
    not*, never as what the stretch is about: the first is arithmetic over the
    transcript and the second is a reading, and only the reader (or the narrator
    in :mod:`modules.advisor`, which is labelled as speculating) gets to make
    it.
    """
    from modules.chapter_speech import NEARLY_SILENT_PCT, SPEECH_DROP, SPEECH_LIFT

    lines = []
    share = chapter.get("speech_share_pct")
    if share is None:                    # no transcript reached this run
        return lines
    share = float(share)
    words = int(chapter.get("words") or 0)

    # The change against the previous chapter first: it is the one speech fact
    # that describes a boundary rather than a stretch, and a chapter list is
    # read top to bottom.
    change = chapter.get("speech_change") or {}
    if change:
        was, now = float(change["from_pct"]), float(change["to_pct"])
        if change["direction"] == "rose":
            lines.append(f"The talking starts here — {was:.0f}% of the previous "
                         f"chapter was speech against {now:.0f}% of this one.")
        else:
            lines.append(f"The talking stops here — {was:.0f}% of the previous "
                         f"chapter was speech against {now:.0f}% of this one.")

    if share <= NEARLY_SILENT_PCT:
        # Said plainly rather than as a percentage: "4% speech" invites the
        # reader to picture sparse dialogue, and four per cent of nine minutes
        # is a handful of words in a stretch that is otherwise silent.
        if not change:
            lines.append("Almost nothing is said in this stretch."
                         if words else "Nothing is said in this stretch.")
    else:
        lift = float(chapter.get("speech_lift") or 0.0)
        rate = float(chapter.get("words_per_minute") or 0.0)
        detail = f"{share:.0f}% of it is speech, {rate:.0f} words a minute"
        if lift >= SPEECH_LIFT:
            lines.append(f"Talks more than the rest of the video — {detail}, "
                         f"{lift:.1f}× the video's rate.")
        elif lift and lift <= SPEECH_DROP:
            lines.append(f"Quieter in words than the rest of the video — "
                         f"{detail}, {lift:.1f}× the video's rate.")
        elif not change:
            lines.append(f"{detail.capitalize()}.")

    found = chapter.get("speech_words") or []
    if found:
        said = ", ".join(str(f["word"]) for f in found)
        # "and the others did not" is load-bearing. Without it the list reads as
        # the chapter's subject, which is a claim about meaning that inverse
        # document frequency cannot support.
        lines.append(f"Words used here that the other chapters did not: {said}.")

    speakers = chapter.get("speakers") or []
    if len(speakers) > 1:
        top = speakers[0]
        lines.append(f"{len(speakers)} voices, the longest speaking for "
                     f"{float(top.get('seconds') or 0):.0f}s.")
    elif len(speakers) == 1:
        lines.append("One voice throughout.")

    return lines


def summarise_speech_run(chapters: Sequence[Mapping],
                         video: Optional[Mapping] = None) -> str:
    """One sentence about how speech is distributed across the whole video.

    The fact worth one line is *concentration*: a transcript whose words all sit
    in two of eleven chapters says the video has a spoken section and a wordless
    one, which is a structural finding no visual signal in this repo produces.
    An evenly spoken video says the opposite, and both are more useful than the
    total word count.
    """
    from modules.chapter_speech import NEARLY_SILENT_PCT

    rows = [c for c in (chapters or []) if c.get("speech_share_pct") is not None]
    if not rows:
        return ""
    total_words = sum(int(c.get("words") or 0) for c in rows)
    if not total_words:
        return "Nothing in the video was transcribed as speech."

    spoken = [c for c in rows if float(c.get("speech_share_pct") or 0.0)
              > NEARLY_SILENT_PCT]
    overall = float((video or {}).get("speech_share_pct") or 0.0)
    head = (f"{total_words} words transcribed"
            + (f", {overall:.0f}% of the runtime" if overall else ""))

    if not spoken:
        return f"{head} — none of it concentrated enough to describe a chapter."
    if len(spoken) == len(rows):
        return f"{head}, spread across every chapter."

    # Ranked, because "which stretches" is the question this sentence exists to
    # answer and an unordered list of numbers does not answer it.
    spoken.sort(key=lambda c: -float(c.get("speech_share_pct") or 0.0))
    named = ", ".join(str(c.get("title") or f"chapter {c.get('number')}")
                      for c in spoken[:3])
    return (f"{head}, concentrated in {len(spoken)} of {len(rows)} chapters "
            f"({named}) — the rest is close to wordless.")


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


# How far a clip's expression reading has to sit from the video's own mean
# before the difference is signed rather than called level. The same bar
# `describe_segment_reading` uses for "in line with the rest of the video", on a
# -1..+1 scale.
READING_DEADBAND = 0.15


def compare_to_video(entry: Mapping,
                     reading: Optional[Mapping] = None) -> list:
    """This clip against the rest of its video, one signed row per axis.

    Everything here is already stated in prose further up the clip. What this
    adds is a shape a reader can take in without reading: five clips scanned
    down the page, and the one that runs against the file is the one with the
    signs that differ from its neighbours'.

    ``+`` and ``-`` mean above and below *this video's own norm*, never good and
    bad. Loudness has no better — a quiet clip in a loud film is as much a
    finding as the reverse — and the expression axis is signed by valence, which
    is a property of the labels a classifier assigned and not of anyone's
    experience.

    Two axes are deliberately absent. Score and the signal percentiles are what
    *selected* the clip, so every kept clip is above the video on them by
    construction; printing a row of plus signs that cannot come out any other
    way would teach a reader to ignore the whole strip. What is here is measured
    independently of the scoring, so a minus is possible on every row.
    """
    rows = []

    loud = entry.get("loudest") or {}
    if loud.get("vs_video_db") is not None:
        db = float(loud["vs_video_db"])
        rows.append({
            "sign": "+" if db >= ABOVE_DB else ("-" if db <= -ABOVE_DB else "="),
            "name": "loudness",
            "figure": f"{db:+.0f} dB on the video's median",
        })

    if reading and reading.get("delta") is not None:
        delta = float(reading["delta"])
        rows.append({
            "sign": ("+" if delta >= READING_DEADBAND
                     else ("-" if delta <= -READING_DEADBAND else "=")),
            "name": "expression reading",
            "figure": f"{delta:+.2f} valence on the video's mean"
                      + (f", mostly {reading['dominant']}"
                         if reading.get("dominant") else ""),
        })

    subject = _leading_subject(entry)
    if subject:
        name, percentile = subject
        rows.append({
            "sign": ("+" if percentile >= NOTABLE
                     else ("-" if percentile <= ORDINARY else "=")),
            "name": f"{name} on screen",
            "figure": f"{percentile:.0f}th percentile for size in this video",
        })
    return rows


def _leading_subject(entry: Mapping) -> Optional[tuple]:
    """The one detected class whose size claim rests on the most, as ``(name, pct)``.

    A relative reading is preferred over a bare frame share for the reason
    :func:`_subject_line` prefers it: moving the camera scales both boxes and
    leaves the ratio alone, where frame share cannot tell a larger subject from
    a closer camera. One row either way — a strip with a line per detected class
    is a table, and a table is the thing this is meant to save the reader from.
    """
    comparison = (entry.get("measured") or {}).get("comparison") or {}
    best = None
    for subject in comparison.get("subjects") or ():
        relative = subject.get("relative") or {}
        if relative.get("enough_samples"):
            rank = float(relative.get("percentile") or 0.0)
            weight = 2
        elif subject.get("enough_samples"):
            rank = float(subject.get("frame_share_percentile") or 0.0)
            weight = 1
        else:
            continue
        # Furthest from the middle, not highest: a clip where the subject is
        # unusually small is exactly as much of a finding as one where it is
        # unusually large, and ranking on the raw percentile would only ever
        # surface the second kind.
        key = (weight, abs(rank - 50.0))
        if best is None or key > best[0]:
            best = (key, str(subject.get("name") or ""), rank)
    if not best or not best[1]:
        return None
    return best[1], best[2]


def format_comparison(rows: Sequence[Mapping]) -> str:
    """The signed rows as one line, for the text report."""
    if not rows:
        return ""
    return "vs the video: " + "  ".join(
        f"{r['sign']} {r['name']} ({r['figure']})" for r in rows)


# Two marked seconds this close are the same event seen twice, not one following
# another. Below it "came before" is a claim about detector timing, not content.
TOGETHER_SECONDS = 2

# A gap wider than this is two things that happened in the same clip, not a
# sequence -- saying one followed the other would invent a link across half a
# minute of unexamined footage.
SEQUENCE_SECONDS = 20


def describe_expression_peak(entry: Mapping) -> str:
    """Where the expression reading settled inside this clip, and on what.

    Says what the classifier reported and how much it had to go on -- the run
    length and the readable-second count are in the sentence because they are
    what separates a reading from a flicker, and a reader who is given only a
    label and a timestamp has no way to tell those apart.

    Phrased as the classifier's reading throughout, for the reasons set out in
    :mod:`modules.expression_arc`: five coarse classes, no notion of intensity,
    and no way to tell a performed expression from a felt one.
    """
    peak = entry.get("expression_peak") or {}
    label = str(peak.get("label") or "")
    if not label or peak.get("second") is None:
        return ""
    stamp = str(peak.get("timestamp", ""))
    held = int(peak.get("seconds") or 0)
    read = int(peak.get("read_seconds") or 0)
    confidence = float(peak.get("confidence") or 0.0)

    if peak.get("turned") and peak.get("from_label"):
        lead = (f"The classifier's reading turns from {peak['from_label']} to "
                f"{label} at {stamp}")
    else:
        lead = f"The classifier reads {label} from {stamp}"
    return (f"{lead}, holding {held}s at {confidence:.2f} — out of {read} "
            f"readable second{'' if read == 1 else 's'} in this clip.")


def describe_reading_shot(entry: Mapping) -> str:
    """Whether the reading changed within a shot, or when the shot changed.

    The single most important qualification the expression channel has, and the
    one that decides whether a turn is worth anything at all.

    A turn inside one continuous shot is the same camera, the same framing and
    the same lighting from one second to the next, so something in the picture
    changed — most plausibly the face, which is the only thing the classifier is
    looking at. That is genuinely stronger evidence than any label or confidence
    figure, because it removes the explanation that competes with all of them.

    A turn on a cut is worth close to nothing. The framing, the angle, the
    lighting and often the subject all change in one frame, and a classifier
    trained on posed frontal faces will report a different label for the same
    person seen from a new angle. Nothing here can separate that from a face
    that changed, so the sentence says so rather than leaving the reader to
    take a coincidence for a finding.

    Silence when no cuts were supplied. "No shot change was detected" and "no
    shot detection ran" read identically and mean opposite things.
    """
    peak = entry.get("expression_peak") or {}
    if not peak.get("label") or peak.get("second") is None:
        return ""
    on_cut = peak.get("at_cut")
    if on_cut is None:
        return ""
    if on_cut:
        return ("That turn lands on a shot change, so the framing, the angle "
                "and the lighting all changed with it — nothing here can "
                "separate a face that changed from a camera that did.")
    return ("It turns inside one continuous shot, with no cut near it: the "
            "framing did not change, so whatever moved the reading was in the "
            "picture rather than in the edit.")


def describe_arrival_shot(entry: Mapping) -> str:
    """Whether something arriving on screen arrived, or was merely framed."""
    onset = entry.get("event_onset") or {}
    name = str(onset.get("name") or "")
    if not name or onset.get("at_cut") is None:
        return ""
    if onset.get("at_cut"):
        return (f"{name} first appears on a shot change, so it may have been "
                f"there before the camera moved to it.")
    return (f"{name} appears without a cut, so it came into a frame that was "
            f"already running.")


def _expression_relation(entry: Mapping) -> str:
    """Where the reading settled relative to the clip's other marked second.

    The loudest point is preferred as the anchor over the motion peak because it
    is the one measured on every clip that has audio, so the ordering stays
    comparable from clip to clip -- which is what
    :func:`summarise_signal_relations` needs to say anything at all.

    Order and distance, nothing else. A reading that arrives after the loudest
    point looks exactly like a reaction and is not evidence of one: the same two
    timestamps are produced by a cut to a different face, by a classifier
    recovering from a blurred frame, and by someone reacting to something.
    """
    peak = entry.get("expression_peak") or {}
    label = str(peak.get("label") or "")
    if not label or peak.get("second") is None:
        return ""

    loud = entry.get("loudest") or {}
    motion = entry.get("motion_peak") or {}
    if loud.get("second") is not None:
        name, at = "the loudest point", int(loud["second"])
    elif motion.get("second") is not None:
        name, at = "the motion peak", int(motion["second"])
    else:
        return ""

    gap = int(peak["second"]) - at
    if abs(gap) > SEQUENCE_SECONDS:
        return ""
    lead = ("The reading turns" if peak.get("turned")
            else "The reading settles")
    if abs(gap) <= TOGETHER_SECONDS:
        return f"{lead} to {label} within {abs(gap)}s of {name}."
    if gap > 0:
        return f"{lead} to {label} {gap}s after {name}."
    return f"{lead} to {label} {abs(gap)}s before {name}."


def _sequence_marks(entry: Mapping) -> list:
    """Every second this clip named, as ``(second, clause, timestamp)``.

    The clause is the mark said as an event rather than as a measurement --
    "movement drops away" rather than "motion peak" -- because this is the one
    line on the clip that is read as a description of what happened, and a row
    of detector names is not one.

    Ties keep detection order rather than being broken arbitrarily; two marks in
    the same second are reported as simultaneous either way.
    """
    marks = []
    onset = entry.get("event_onset") or {}
    if onset.get("second") is not None and onset.get("name"):
        marks.append((int(onset["second"]), f"{onset['name']} comes on screen",
                      str(onset.get("timestamp", "")), "event"))
    motion = entry.get("motion_peak") or {}
    if motion.get("second") is not None:
        marks.append((int(motion["second"]), "movement drops away",
                      str(motion.get("timestamp", "")), "motion"))
    loud = entry.get("loudest") or {}
    if loud.get("second") is not None:
        marks.append((int(loud["second"]), "the loudest point arrives",
                      str(loud.get("timestamp", "")), "loudness"))
    reading = entry.get("expression_peak") or {}
    label = str(reading.get("label") or "")
    if reading.get("second") is not None and label:
        verb = "turns to" if reading.get("turned") else "settles on"
        marks.append((int(reading["second"]), f"the reading {verb} {label}",
                      str(reading.get("timestamp", "")), "expression"))
    marks.sort(key=lambda m: m[0])
    return marks


def _longest_chain(marks: Sequence[tuple]) -> list:
    """The longest stretch of marks with no unexamined gap between them.

    A chain is only a chain while each step follows the one before closely
    enough to be the same passage of footage. Half a minute of unexamined video
    between two marks makes them two things that shared a clip, and stringing
    them into one sentence would invent the connection the sentence is being
    read for.
    """
    best: list = []
    current: list = []
    for mark in marks:
        if current and mark[0] - current[-1][0] > SEQUENCE_SECONDS:
            current = []
        current = current + [mark]
        if len(current) > len(best):
            best = current
    return best


def describe_sequence(entry: Mapping) -> str:
    """What happened in this clip, in the order it happened.

    The pairwise sentences below are accurate and are hard work to read: three
    marks produce three comparisons, and a reader assembles the actual sequence
    from them by hand. Once all three exist the sequence *is* the finding, and
    it costs nothing to state -- every second in it was already measured.

    "In clock order" is not decoration. This sentence reads as a story, which is
    what makes it worth having and also what makes it dangerous: a reader takes
    a story to be causal unless told otherwise, and nothing here can separate
    "the movement stopped and she reacted" from a cut to a different face, a
    classifier recovering from a blurred frame, or two unrelated things ten
    seconds apart. Three words at the front frame the whole line as an ordering,
    every time, without hedging any clause inside it.
    """
    marks = _longest_chain(_sequence_marks(entry))
    # Three marks make a sequence worth telling as one. Two are normally left to
    # the pairwise comparisons, which name both timestamps -- except when one of
    # them is an arrival, because that mark has no comparison of its own and the
    # sequence is the only place it can be said at all.
    carries_arrival = any(m[3] == "event" for m in marks)
    if len(marks) < (2 if carries_arrival else 3):
        return ""

    clauses = [f"{marks[0][1]} at {marks[0][2]}"]
    for i, (second, clause, _stamp, _kind) in enumerate(marks[1:], start=1):
        gap = second - marks[i - 1][0]
        if gap == 0:
            clauses.append(f"{clause} in the same second")
        elif gap <= TOGETHER_SECONDS:
            clauses.append(f"{clause} {gap}s after")
        elif i == 1:
            clauses.append(f"{clause} {gap}s later")
        else:
            clauses.append(f"{clause} {gap}s after that")
    return f"In order: {_join(clauses)}."


def describe_signal_relations(entry: Mapping) -> str:
    """How the seconds this clip named relate to each other.

    The report marks a loudest second, a motion peak and the second the
    expression reading settled, and left to itself would leave them as three
    unconnected facts. Their *order* is the thing a reader reconstructs by hand
    otherwise, and it is free: all three are already measured.

    Says which came first and how far apart, and nothing about why. One clip
    cannot distinguish "the movement stopped and then she reacted" from two
    unrelated events sharing thirty seconds, and the sentence must not pretend
    otherwise -- :func:`summarise_signal_relations` is where a repeated ordering
    becomes evidence, because that needs the whole run.

    With all three marks in one passage the sequence replaces the comparisons
    rather than joining them: the same three facts stated once in order are
    shorter than three pairwise sentences and are what a reader was assembling
    from them anyway. Below three, the comparisons stay -- naming both
    timestamps is worth more than a chain of two links.
    """
    sequence = describe_sequence(entry)
    if sequence:
        return sequence
    said = [_loudness_motion_relation(entry), _expression_relation(entry)]
    return " ".join(s for s in said if s)


def _loudness_motion_relation(entry: Mapping) -> str:
    """The original pair: where the loudest point sat against the motion peak."""
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


# Counts as a person says them. The conclusion carries no figures at all -- the
# tiles at the top of the page and every section below it are full of them, and
# a paragraph meant to be read as a description stops being one the moment it
# asks the reader to compare numbers.
_COUNT_WORDS = ("no", "one", "two", "three", "four", "five", "six", "seven",
                "eight", "nine", "ten", "eleven", "twelve")


def _count_word(n: int) -> str:
    n = int(n or 0)
    return _COUNT_WORDS[n] if n < len(_COUNT_WORDS) else "several"


def _share_words(count: int, total: int) -> str:
    """"in nearly every clip" and friends, from a proportion."""
    if not total:
        return ""
    share = count / total
    if share >= 0.9:
        return "in nearly every clip"
    if share >= PATTERN_AGREEMENT:
        return "in most clips"
    if share >= 0.4:
        return "in about half the clips"
    return "in some clips"


def _gap_words(seconds: float) -> str:
    """A lag in the words a person uses for it, not in seconds."""
    seconds = abs(float(seconds or 0))
    if seconds <= TOGETHER_SECONDS:
        return "at the same moment"
    if seconds <= 5:
        return "a moment later"
    if seconds <= 12:
        return "a few seconds later"
    return "several seconds later"


def conclude(report: Mapping) -> list:
    """What the run found, as headed sections of plain description.

    Everything here is already stated in figures somewhere in the report. That
    is the point rather than a weakness: the findings are spread over a page of
    clips, a chapter table and four summaries, and a reader who wants "so what
    is this video, and what did the run make of it" has to hold all of them at
    once to answer it.

    So this carries no numbers. Not as decoration -- a description a reader has
    to assemble out of percentages is not a description, and every figure behind
    these sentences is a few inches away in the section it came from. What is
    left is the shape: which signals agreed, in what order, and how often.

    Returns ``[{"heading": ..., "lines": [...]}, ...]``, sections only where a
    measurement exists to fill them. A run with no faces, no chapters and no
    categories still concludes -- with fewer sections, and without reaching for
    the ones it has no evidence for.
    """
    sections = []
    for heading, builder in (
        ("The video", _section_video),
        ("Movement", _section_movement),
        ("Sound", _section_sound),
        ("Face expression", _section_expression),
        ("On screen", _section_on_screen),
    ):
        lines = [line for line in builder(report) if line]
        if lines:
            sections.append({"heading": heading, "lines": lines})
    if not sections:
        return []

    summary = _section_summary(report)
    if summary:
        sections.append({"heading": "Summary", "lines": summary})
    return sections


def _section_video(report: Mapping) -> list:
    """What the file is, and which parts of it the cut drew on."""
    chapters = list(report.get("chapters") or [])
    lines = []

    weighted: dict = {}
    for chapter in chapters:
        dominant = chapter.get("dominant") or {}
        name = str(dominant.get("name") or "")
        if name:
            weighted[name] = weighted.get(name, 0.0) + float(
                chapter.get("duration") or 0.0)

    if len(chapters) > 1:
        said = f"The video divides into {_count_word(len(chapters))} stretches"
        if weighted:
            top = max(sorted(weighted), key=lambda k: weighted[k])
            said += f", mostly {top} on screen"
        lines.append(said + ".")
    elif weighted:
        top = max(sorted(weighted), key=lambda k: weighted[k])
        lines.append(f"Mostly {top} on screen throughout.")

    used = [c for c in chapters if int(c.get("clips") or 0)]
    if len(chapters) > 1 and used:
        lead = max(used, key=lambda c: float(c.get("cut_share_pct") or 0.0))
        where = ("one of them" if len(used) == 1
                 else f"{_count_word(len(used))} of them")
        lines.append(f"The cut came from {where}, most of it from "
                     f"{lead.get('title', 'a single stretch')}.")
    return lines


def _section_movement(report: Mapping) -> list:
    """Whether the kept moments have the burst-then-stillness shape, and where."""
    segments = list(report.get("segments") or [])
    marked = [e for e in segments
              if (e.get("motion_peak") or {}).get("second") is not None]
    if not marked:
        return []

    lines = [f"Movement bursts and then settles "
             f"{_share_words(len(marked), len(segments))}."]
    found = _pattern(_mark_gaps(segments, "loudest", "motion_peak"))
    if found and found["kind"] == "after":
        lines.append("It settles before the loudest point rather than during "
                     "it — the sound arrives once the movement has stopped.")
    elif found and found["kind"] == "before":
        lines.append("It settles after the loudest point, so the sound comes "
                     "first.")
    elif found and found["kind"] == "together":
        lines.append("It settles as the loudest point arrives, not before it.")
    return lines


def _section_sound(report: Mapping) -> list:
    """How loud the kept moments are for this video, and what they sound like."""
    segments = list(report.get("segments") or [])
    deltas = [float((e.get("loudest") or {}).get("vs_video_db"))
              for e in segments
              if (e.get("loudest") or {}).get("vs_video_db") is not None]
    lines = []
    if deltas:
        middle = sorted(deltas)[len(deltas) // 2]
        phrase = _level_phrase(middle)
        if phrase == "close to":
            lines.append("The kept moments sit about as loud as the video "
                         "usually is — loudness is not what marks them out.")
        elif phrase == "below":
            lines.append("The kept moments are quieter than the video usually "
                         "is.")
        else:
            lines.append(f"The kept moments are {phrase} the video's usual "
                         f"level.")

    comparison = (report.get("level_by_class") or {}).get("comparison") or {}
    if comparison.get("resolvable"):
        lines.append(f"Across the file it is consistently louder while "
                     f"{comparison.get('louder')} is on screen than while "
                     f"{comparison.get('quieter')} is.")

    lines.append(_peak_candidate(report))
    return [line for line in lines if line]


def _peak_candidate(report: Mapping) -> str:
    """The single loudest moment in the file, offered as a place to look.

    A ranking, not an identification, and the difference is the whole sentence.
    Asked "where does this video peak", the level curve has exactly one answer
    and it is a defensible one: of everything measured here, this is the moment
    the file is loudest. What that moment *is* — the point a scene builds to,
    an edit artefact, a door slamming — level cannot say, and the report would
    be inventing the interesting half of the answer if it tried.

    Worth stating anyway, because "the top of this measurement" is exactly what
    someone looking for the peak of a video should check first, and finding it
    by hand means scrubbing the whole file.
    """
    loudest = (report.get("level_by_class") or {}).get("loudest") or {}
    stamp = str(loudest.get("timestamp") or "")
    if not stamp:
        return ""
    classes = [str(c) for c in (loudest.get("classes") or ())]
    where = f", with {_join(classes)} on screen" if classes else ""
    return (f"Its single loudest moment is at {stamp}{where}. If you are "
            f"looking for the point the video builds to, that is the first "
            f"place to check — it is the top of one measurement, not a finding "
            f"about what happens there.")


def _section_expression(report: Mapping) -> list:
    """How much of the file could be read, and what the reading did."""
    arc = report.get("expression_arc") or {}
    if not arc:
        return []
    lines = []

    coverage = float((arc.get("coverage") or {}).get("pct") or 0.0)
    if coverage >= 70:
        lines.append("A face was readable through most of the video.")
    elif coverage >= 35:
        lines.append("A face was readable through about half the video, so "
                     "this describes that half.")
    else:
        lines.append("A face was readable through only part of the video, so "
                     "this describes that part and not the rest.")

    shift = arc.get("shift") or {}
    trend = arc.get("arc") or {}
    if shift:
        lines.append(f"The reading turns "
                     f"{str(shift['direction']).replace('toward ', 'more ')} "
                     f"partway through, around {_clock(shift['at'])}.")
    elif trend.get("confident"):
        lines.append(f"The reading drifts "
                     f"{str(trend['direction']).replace('toward ', 'more ')} "
                     f"across the file.")

    found = _pattern(_mark_gaps(report.get("segments") or [],
                                "expression_peak", "loudest"))
    if found and found["kind"] == "after":
        lines.append(f"Inside the kept clips it settles after the loudest "
                     f"point, {_gap_words(found['median'])}.")
    elif found and found["kind"] == "before":
        lines.append("Inside the kept clips it is already in place before the "
                     "loudest point.")
    return lines


def _section_on_screen(report: Mapping) -> list:
    """The category the kept clips have in common, if they have one."""
    segments = list(report.get("segments") or [])
    names: dict = {}
    for entry in segments:
        name = str((entry.get("event_onset") or {}).get("name") or "")
        if name:
            names[name] = names.get(name, 0) + 1
    if not names:
        return []
    top = max(sorted(names), key=lambda k: names[k])
    if names[top] < MIN_CLIPS_FOR_PATTERN:
        return []
    # The share phrase leads, so the sentence does not have to say "clip"
    # twice — and so no detected name has to start a sentence and be
    # sentence-cased, which would rewrite what the detector produced.
    share = _share_words(names[top], len(segments))
    return [f"{share[0].upper()}{share[1:]}, {top} comes on screen partway "
            f"through rather than being there when the clip starts."]


def _section_summary(report: Mapping) -> list:
    """The whole run as one sentence a person would say.

    This is the line the rest of the report exists to support, and the line
    where the temptation to say more than was measured is strongest. It states
    the order the marks arrive in, in the words someone would use for it, and
    then says what that order is and is not -- because a sequence read as a
    story is read as a cause, and no arrangement of these timestamps can tell
    one from the other.

    What it never does is name a feeling. The expression channel reports the
    label a five-class classifier put on a face, and a loud second is a loud
    second: neither can separate a felt reaction from a performed one, and that
    separation is the whole of the claim a reader would want made here.
    """
    segments = list(report.get("segments") or [])
    if not segments:
        return []

    steps = []
    names: dict = {}
    for entry in segments:
        name = str((entry.get("event_onset") or {}).get("name") or "")
        if name:
            names[name] = names.get(name, 0) + 1
    if names:
        top = max(sorted(names), key=lambda k: names[k])
        if names[top] >= MIN_CLIPS_FOR_PATTERN:
            steps.append(f"{top} comes on screen")

    sound = _pattern(_mark_gaps(segments, "loudest", "motion_peak"))
    if sound and sound["kind"] == "after":
        steps.append("movement settles")
        steps.append(f"the loudest moment follows {_gap_words(sound['median'])}")
    elif sound and sound["kind"] == "before":
        steps.append("the loudest moment arrives")
        steps.append("movement settles after it")
    elif sound and sound["kind"] == "together":
        steps.append("movement settles and the loudest moment arrives together")

    reading = _pattern(_mark_gaps(segments, "expression_peak", "loudest"))
    if reading and reading["kind"] in ("after", "before"):
        labels: dict = {}
        for entry in segments:
            label = str((entry.get("expression_peak") or {}).get("label") or "")
            if label:
                labels[label] = labels.get(label, 0) + 1
        label = (max(sorted(labels), key=lambda k: labels[k]) if labels else "")
        if label and reading["kind"] == "after":
            steps.append(f"and the face reads {label} just after")
        elif label:
            steps.append(f"with the face already reading {label} before it")

    if len(steps) < 2:
        return []
    # One short line, not the paragraph this used to be. The limit has to be
    # visible, and a reader who meets four sentences of it under every report
    # stops reading the section rather than the caveat — which loses both. The
    # sections above already say what each measurement is; this only has to say
    # that an order is not a cause.
    return [
        "Put together, most of the kept clips run the same way: "
        + ", ".join(steps)
        + " — an order, not a cause; what it meant is for whoever watches it.",
    ]


def _mark_gaps(segments: Sequence[Mapping], later: str, earlier: str) -> list:
    """Signed distances between two marked seconds, one per clip that has both.

    Clips where the two are further apart than a sequence can span are dropped
    rather than counted: they are two things that shared a clip, and including
    them would let unrelated pairs vote on an ordering.
    """
    gaps = []
    for e in segments or ():
        a = (e.get(later) or {}).get("second")
        b = (e.get(earlier) or {}).get("second")
        if a is None or b is None:
            continue
        gap = int(a) - int(b)
        if abs(gap) <= SEQUENCE_SECONDS:
            gaps.append(gap)
    return gaps


def _pattern(gaps: Sequence[int]) -> dict:
    """Which side of one mark the other tends to land on, across the run.

    Split out so the long sentence and the one-line conclusion are computed
    once. Two prose functions deriving the same finding separately is how a
    report comes to say "in 7 of 9 clips" in one place and something else three
    inches further down.
    """
    if len(gaps) < MIN_CLIPS_FOR_PATTERN:
        return {}
    groups = {
        "after": [g for g in gaps if g > TOGETHER_SECONDS],
        "before": [g for g in gaps if g < -TOGETHER_SECONDS],
        "together": [g for g in gaps if abs(g) <= TOGETHER_SECONDS],
    }
    total = len(gaps)
    for kind, group in groups.items():
        if group and len(group) / total >= PATTERN_AGREEMENT:
            median = sorted(abs(g) for g in group)[len(group) // 2]
            return {"kind": kind, "count": len(group), "total": total,
                    "median": median}
    return {"kind": "none", "total": total,
            **{k: len(v) for k, v in groups.items()}}


def _expression_pattern(segments: Sequence[Mapping]) -> str:
    """Whether the reading lands on the same side of the loudest point every time.

    The per-clip sentence can only report an order. Repetition is what turns it
    into something about the footage: one clip where the reading arrives two
    seconds after the loudest point is a coincidence, eight of eleven is a
    property of how this material was shot or cut -- still not a cause, and the
    sentence says so, because a consistent lag is exactly what a consistent
    editing rhythm produces as well.
    """
    found = _pattern(_mark_gaps(segments, "expression_peak", "loudest"))
    if not found:
        return ""
    total = found["total"]

    tail = ("What that ordering means is not something these timestamps can "
            "settle — a lag that repeats is as much a property of how the "
            "footage was cut as of anything in it.")
    if found["kind"] == "after":
        return (f"The expression reading settles after the loudest point in "
                f"{found['count']} of {total} clips, typically about "
                f"{found['median']}s later. {tail}")
    if found["kind"] == "before":
        return (f"The expression reading is already in place before the loudest "
                f"point in {found['count']} of {total} clips, by about "
                f"{found['median']}s. {tail}")
    if found["kind"] == "together":
        return (f"The expression reading and the loudest point land within "
                f"{TOGETHER_SECONDS}s of each other in {found['count']} of "
                f"{total} clips.")
    return (f"The expression reading falls on no consistent side of the loudest "
            f"point — {found['after']} of {total} clips after it, "
            f"{found['before']} before, {found['together']} together.")


# At most this many categories get a paragraph. The list is sorted by how many
# clips carry each, so the cut-off keeps the ones with evidence behind them --
# and a report that profiles every class detected is a table again.
MAX_EVENT_SUMMARIES = 3


def summarise_event_relations(segments: Sequence[Mapping],
                              readings: Optional[Mapping] = None) -> list:
    """What tends to happen around each category, counted across the run.

    This is the closest the report gets to answering "what does X mean in this
    video", and the form of the answer is a count. Three clips where the loudest
    point follows a category is an anecdote; six of seven is a regularity in
    this file, and a reader can act on the second without being told what it
    means.

    Two things it will not do, both for the same reason -- they are not in the
    measurements:

    It does not say one thing caused another. Every figure here is a
    co-occurrence: the marks arrive together, and cutting, camera work, and what
    the material is all produce that too.

    It does not convert the expression reading into an experience. The reading
    is the label a five-class classifier assigned to a face, it cannot separate
    a performed expression from a felt one, and material where people are
    performing is exactly where that distinction decides the answer. Counting
    how often the reading runs positive is a fact about the labels; what it
    reflects is the reader's call, made with the clip in front of them.
    """
    by_name: dict = {}
    for entry in segments or ():
        onset = entry.get("event_onset") or {}
        name = str(onset.get("name") or "")
        if name:
            by_name.setdefault(name, []).append(entry)

    lines = []
    ordered = sorted(by_name.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    for name, clips in ordered[:MAX_EVENT_SUMMARIES]:
        if len(clips) < MIN_CLIPS_FOR_PATTERN:
            continue
        line = _event_summary(name, clips, readings or {})
        if line:
            lines.append(line)
    return lines


def _event_summary(name: str, clips: Sequence[Mapping],
                   readings: Mapping) -> str:
    """One category's line: how many clips, and what the other marks did in them."""
    lags = []
    for entry in clips:
        onset = int((entry.get("event_onset") or {})["second"])
        loud = (entry.get("loudest") or {}).get("second")
        if loud is None:
            continue
        gap = int(loud) - onset
        if 0 <= gap <= SEQUENCE_SECONDS:
            lags.append(gap)

    positive = negative = level = 0
    for entry in clips:
        row = readings.get(entry.get("index")) or {}
        delta = row.get("delta")
        if delta is None:
            continue
        if float(delta) >= READING_DEADBAND:
            positive += 1
        elif float(delta) <= -READING_DEADBAND:
            negative += 1
        else:
            level += 1

    # Led by a fixed phrase so no detected name ever has to start a sentence --
    # sentence-casing one would rewrite the data the detector produced.
    parts = [f"Across the run, {name} arrives in {len(clips)} of the kept clips"]
    if lags:
        median = sorted(lags)[len(lags) // 2]
        # "typically about", not "within": the median is the middle of the lags,
        # so half of them are longer than it and "within" would be false for
        # those clips.
        parts.append(f"the loudest point follows it in {len(lags)} of them, "
                     f"typically about {median}s later")
    read = positive + negative + level
    if read:
        if positive > negative and positive:
            parts.append(f"the expression reading runs more positive than the "
                         f"video's own in {positive} of the {read} with a "
                         f"readable face")
        elif negative > positive:
            parts.append(f"the expression reading runs more negative than the "
                         f"video's own in {negative} of the {read} with a "
                         f"readable face")
        else:
            parts.append(f"the expression reading is level with the video's "
                         f"own in most of the {read} with a readable face")
    if len(parts) == 1:
        return ""
    return (_join(parts)
            + ". Counted on this video: the marks arrive together, which is "
              "not the same as one producing another, and the reading is a "
              "classifier's label rather than anyone's experience.")


def summarise_signal_relations(segments: Sequence[Mapping]) -> str:
    """Whether the ordering repeats across the run, which is what makes it a finding.

    One clip where movement stopped nine seconds before the loudest point is a
    coincidence. Seven of nine is a property of the footage, and it is the kind
    of thing a person would otherwise only notice after watching the whole
    thing twice. Returns nothing when the clips disagree -- a split verdict is
    the honest output of a split measurement.

    Each pair of marks is judged on its own clips, because they are not measured
    on the same ones: a clip with no readable face still has a loudest second,
    and pooling them would let the pair with more evidence speak for the other.
    """
    said = [_loudness_motion_pattern(segments), _expression_pattern(segments)]
    return " ".join(s for s in said if s)


def _loudness_motion_pattern(segments: Sequence[Mapping]) -> str:
    """The original pair, across the run: sound against movement stopping."""
    found = _pattern(_mark_gaps(segments, "loudest", "motion_peak"))
    if not found:
        return ""
    total = found["total"]

    if found["kind"] == "after":
        return (f"In {found['count']} of {total} clips the loudest point "
                f"arrives after movement has stopped, typically about "
                f"{found['median']}s later. That ordering is a property of this "
                f"footage, not of any one clip — what it means is still a "
                f"matter for whoever watches it.")
    if found["kind"] == "before":
        return (f"In {found['count']} of {total} clips the loudest point comes "
                f"first and movement drops away about {found['median']}s "
                f"afterwards.")
    if found["kind"] == "together":
        return (f"In {found['count']} of {total} clips the loudest point and "
                f"the motion peak land within {TOGETHER_SECONDS}s of each "
                f"other.")
    return (f"Across {total} clips the two land in no consistent order — "
            f"{found['after']} with sound after movement, {found['before']} "
            f"before, {found['together']} together. Nothing here links them.")
