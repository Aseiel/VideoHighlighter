"""How the expression reading moves across a video, and whether to believe it.

The per-second scan answers "what was the face doing here". Stacked over a whole
file it can answer something people actually want: does the reading *change* —
negative early and positive later, one stretch unlike the rest, a run of clips
that read differently from the video around them.

That is a genuinely useful question and this module answers it. What it will not
do is convert the answer into a claim about how anyone felt, and the distinction
is not squeamishness — it is the difference between a number that holds up and
one that does not:

* The classifier has five coarse classes, no notion of intensity, and returns a
  label for every face it is handed. It was trained on posed, frontal, well-lit
  faces and degrades on profile, occlusion, motion blur and unusual lighting.
* A facial expression is not an emotion. The same face reads as several things
  depending on context the model cannot see, and the mapping from expression to
  felt experience is contested in the research literature rather than settled.
* Expressions in recorded footage are frequently *performed*. A model cannot
  separate a performed expression from a felt one, because there is nothing in
  the pixels that distinguishes them.

So everything here is phrased, named and structured as a property of *the
reading*: `valence` is the reading's valence, an arc is an arc in what the
classifier reported. That framing is what makes the output safe to act on — a
user hunting for the stretches that read negative gets exactly that, and is
never handed a verdict the pixels cannot support.

The most important thing the module computes is the reason to distrust itself.
Three measurements exist for that:

* **Coverage** — a reading over 12% of a file describes 12% of a file. Every
  share here is a share of what was *read*, and the coverage figure is what
  converts it back into a share of the video.
* **Stability** — how often the label changes between one read second and the
  next. A classifier flipping every second is not tracking anything, and its
  "arc" is a smoothing artefact.
* **Dispersion** — whether a label clusters or is smeared evenly across the
  whole file. This is the decisive one. A model that systematically misreads a
  kind of footage produces its favourite label *uniformly*; a real change in
  what is happening produces it in *stretches*. A label that dominates a video
  and is spread flat through it is the signature of a bias, and saying so is
  worth more than any arc fitted through it.
"""
from __future__ import annotations

import math
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

# What each built-in class contributes to the reading's valence. `surprise` is
# deliberately unvalenced: it is the one class of the five that is genuinely
# ambiguous in sign — delight and alarm produce it alike — and folding it into
# either side would let the largest single swing in most files be produced by
# the one label that cannot support a direction.
VALENCE = {
    "happy": 1.0,
    "sad": -1.0,
    "anger": -1.0,
    "neutral": 0.0,
}
UNVALENCED = ("surprise",)

# How many buckets the file is divided into for the arc. Enough that a shift in
# the middle third is visible, few enough that each bucket holds real evidence
# on a video of a few minutes.
DEFAULT_BUCKETS = 12

# A bucket with fewer read seconds than this is not evidence, and is left out of
# the trend fit rather than dragged to zero.
MIN_BUCKET_READS = 3

# Mean valence has to move at least this much between the start and end of the
# fit before the arc is called a direction rather than a flat line. The scale is
# -1..+1, so this is a tenth of the full range.
ARC_CHANGE = 0.20

# How much of the variation a straight line has to explain before the direction
# it points in is worth stating without qualification. A steep line through a
# scatter is a trend the file does not have, and reporting the slope while
# hiding the fit is the most flattering thing this module could do.
GOOD_FIT = 0.35

# A single change point has to move the reading at least this much to be called
# one. Higher than ARC_CHANGE: a split is a stronger claim than a drift, because
# it says the file has two halves rather than one direction.
SHIFT_CHANGE = 0.22

# Buckets either side of a split, below which it is describing an endpoint
# rather than a phase.
MIN_SHIFT_BUCKETS = 2

# Below this share of the video carrying a readable face, every share computed
# here is describing a minority of the file.
LOW_COVERAGE = 35.0

# Mean confidence below this is a classifier picking between near-ties — the
# same floor `face_emotions` uses to decide a face is unreadable.
LOW_CONFIDENCE = 0.6

# Fewer consecutive seconds than this per label run means the reading changes
# faster than anything on screen plausibly does.
UNSTABLE_RUN = 2.0

# A label holding at least this much of the reading is the video's character
# rather than an event in it — and if it is also spread evenly, that is the
# shape of a systematic misread rather than of anything that happened.
DOMINANT_SHARE = 45.0

# Coefficient of variation across buckets, below which a label is spread evenly
# enough to be called uniform. Clustered labels run well above this.
UNIFORM_CV = 0.35

# How far a class's reading has to sit from the video's own baseline before the
# two are called different. On a -1..+1 scale this is a modest bar, and most
# classes will not clear it — which is the finding. "The reading while X is on
# screen is the same as everywhere else" is the answer to the question people
# actually ask here, and it is the answer far more often than not.
CLASS_DELTA = 0.15

# Readable seconds a class needs before its reading is compared at all. A mean
# over eleven seconds is not a property of the class.
MIN_CLASS_SECONDS = 30

# A label held for fewer read seconds than this is a flicker rather than a
# reading. Same number UNSTABLE_RUN calls too fast to be tracking anything: a
# single second unlike the seconds either side of it is what a classifier
# produces on a blurred frame, and pinning a moment on it would put a timestamp
# on noise.
MIN_PEAK_RUN = 2

# How far back a differently-labelled second may sit and still count as what the
# reading turned *from*. Beyond it the face was unreadable in between, so
# "turned" would be describing a gap in the scan rather than a change on screen.
TURN_GAP = 4


def _normalise(expressions: Optional[Mapping]) -> dict:
    """``{second: (label, confidence)}`` from either shape the app produces."""
    out: dict = {}
    for sec, value in (expressions or {}).items():
        if isinstance(value, Mapping):
            label = str(value.get("label") or "")
            confidence = float(value.get("confidence") or 0.0)
        elif isinstance(value, (tuple, list)) and len(value) >= 2:
            label, confidence = str(value[0] or ""), float(value[1] or 0.0)
        else:
            continue
        if label:
            out[int(sec)] = (label, confidence)
    return out


def valence_of(label: str, confidence: float = 1.0) -> float:
    """The reading's valence for one second, weighted by how sure the model was.

    Weighting matters more here than anywhere else in the report: an arc built
    from labels alone treats a 0.51 coin-flip and a 0.97 reading as the same
    evidence, and in footage the classifier finds hard that is most of the
    input.
    """
    return VALENCE.get(str(label).lower(), 0.0) * float(confidence)


def _runs(seconds: Sequence[int], labels: Mapping) -> list:
    """Contiguous runs of one label, as ``(label, start, end, length)``.

    Contiguity is in *read* seconds, not clock seconds: a scan taken every two
    seconds, or one that skipped frames with no face, would otherwise report
    every reading as its own run and call the file maximally unstable.
    """
    out = []
    start = None
    previous_label = None
    previous_sec = None
    for sec in seconds:
        label = labels[sec][0]
        if label != previous_label:
            if previous_label is not None:
                out.append((previous_label, start, previous_sec,
                            previous_sec - start + 1))
            start, previous_label = sec, label
        previous_sec = sec
    if previous_label is not None:
        out.append((previous_label, start, previous_sec, previous_sec - start + 1))
    return out


def _buckets(labels: Mapping, duration: float, count: int) -> list:
    """The file cut into equal spans, each with its own reading."""
    if duration <= 0 or count <= 0:
        return []
    width = duration / count
    out = []
    for i in range(count):
        lo, hi = i * width, (i + 1) * width
        inside = [(label, confidence)
                  for sec, (label, confidence) in labels.items()
                  if lo <= sec < hi]
        entry = {
            "index": i,
            "start": round(lo, 1),
            "end": round(hi, 1),
            "read_seconds": len(inside),
        }
        if inside:
            entry["valence"] = round(
                float(np.mean([valence_of(l, c) for l, c in inside])), 3)
            counts: dict = {}
            for label, _c in inside:
                counts[label] = counts.get(label, 0) + 1
            entry["dominant"] = max(sorted(counts), key=lambda k: counts[k])
            entry["shares"] = {k: round(100.0 * v / len(inside), 1)
                               for k, v in counts.items()}
        out.append(entry)
    return out


def _arc(buckets: Sequence[Mapping]) -> dict:
    """A straight line through the bucket valences, and what it amounts to.

    A line rather than "first third versus last third": thirds are three numbers
    and a shift in any one of them swings the answer, while a fit over every
    bucket with evidence in it uses the whole file. Buckets with too little read
    in them are dropped rather than counted as neutral — a stretch with no face
    is not a stretch of calm.
    """
    usable = [b for b in buckets
              if b.get("read_seconds", 0) >= MIN_BUCKET_READS and "valence" in b]
    if len(usable) < 3:
        return {"direction": "unknown", "reason": "too few readable stretches",
                "buckets_used": len(usable)}

    x = np.asarray([b["index"] for b in usable], dtype=float)
    y = np.asarray([b["valence"] for b in usable], dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    first, last = float(slope * x[0] + intercept), float(slope * x[-1] + intercept)
    change = last - first

    if abs(change) < ARC_CHANGE:
        direction = "flat"
    else:
        direction = "toward positive" if change > 0 else "toward negative"

    # How well the line actually describes the points. A steep fit through a
    # scatter is a direction the file does not have, and the slope alone cannot
    # tell the reader which of the two they are looking at.
    fit = (round(float(np.corrcoef(x, y)[0, 1] ** 2), 3)
           if len(usable) > 2 and y.std() > 0 else 0.0)
    return {
        "direction": direction,
        "change": round(change, 3),
        "start_valence": round(first, 3),
        "end_valence": round(last, 3),
        "buckets_used": len(usable),
        "fit": fit,
        # Whether the direction is worth stating flatly. Kept as its own field
        # so the sentence layer does not have to know what an R² is.
        "confident": bool(direction != "flat" and fit >= GOOD_FIT),
    }


def _shift(buckets: Sequence[Mapping]) -> dict:
    """The single point where the reading changes most, if there is one.

    A straight line can only say "it drifts". The question people actually ask —
    "does it start one way and turn?" — is about a *split*, and a file whose
    first quarter reads flat and whose remainder reads negative has a strong
    answer to it and almost no slope. Both measurements are cheap, and reporting
    only the line would miss the more legible half of the structure.

    Means are weighted by how much was read in each bucket, so a stretch where
    the face was visible for six seconds does not outvote one where it was
    visible for two hundred.
    """
    usable = [b for b in buckets
              if b.get("read_seconds", 0) >= MIN_BUCKET_READS and "valence" in b]
    if len(usable) < MIN_SHIFT_BUCKETS * 2:
        return {}

    values = np.asarray([b["valence"] for b in usable], dtype=float)
    weights = np.asarray([b["read_seconds"] for b in usable], dtype=float)

    best = None
    for cut in range(MIN_SHIFT_BUCKETS, len(usable) - MIN_SHIFT_BUCKETS + 1):
        before = float(np.average(values[:cut], weights=weights[:cut]))
        after = float(np.average(values[cut:], weights=weights[cut:]))
        change = after - before
        if best is None or abs(change) > abs(best[0]):
            best = (change, cut, before, after)
    if best is None or abs(best[0]) < SHIFT_CHANGE:
        return {}

    change, cut, before, after = best
    return {
        "at": round(float(usable[cut]["start"]), 1),
        "before": round(before, 3),
        "after": round(after, 3),
        "change": round(change, 3),
        "direction": "toward positive" if change > 0 else "toward negative",
    }


def _dispersion(buckets: Sequence[Mapping], label: str) -> dict:
    """Whether a label clusters into stretches or is smeared across the file.

    The measurement that decides whether an arc is worth reading at all. A
    classifier that systematically misreads a kind of footage emits its
    favourite label at a steady rate throughout; something that actually
    happened at some point in the video emits it in bursts. Those two look
    identical in a total and completely different here.
    """
    shares = [b.get("shares", {}).get(label, 0.0)
              for b in buckets if b.get("read_seconds", 0) >= MIN_BUCKET_READS]
    if len(shares) < 3:
        return {}
    mean = float(np.mean(shares))
    if mean <= 0:
        return {}
    cv = float(np.std(shares) / mean)
    peak = max(range(len(shares)), key=lambda i: shares[i])
    return {
        "cv": round(cv, 3),
        "uniform": bool(cv < UNIFORM_CV),
        "peak_share_pct": round(max(shares), 1),
        "low_share_pct": round(min(shares), 1),
        "mean_share_pct": round(mean, 1),
    }


def _episodes(labels: Mapping, seconds: Sequence[int], min_seconds: float,
              gap: float) -> list:
    """Stretches that read consistently one way, as time ranges.

    This is the part a user acts on: "which parts read negative" is a list of
    places to look, and a list is worth more than any single verdict about the
    file. Sign rather than label, so a stretch that alternates between the two
    negative classes is one episode instead of six.
    """
    out = []
    current = None
    for sec in seconds:
        label, confidence = labels[sec]
        value = valence_of(label, confidence)
        sign = 0 if value == 0 else (1 if value > 0 else -1)
        if sign == 0:
            continue
        if (current and current["sign"] == sign
                and sec - current["last"] <= gap):
            current["last"] = sec
            current["values"].append(value)
            current["labels"].append(label)
            continue
        if current:
            out.append(current)
        current = {"sign": sign, "start": sec, "last": sec,
                   "values": [value], "labels": [label]}
    if current:
        out.append(current)

    episodes = []
    for run in out:
        span = run["last"] - run["start"] + 1
        if span < min_seconds:
            continue
        counts: dict = {}
        for label in run["labels"]:
            counts[label] = counts.get(label, 0) + 1
        episodes.append({
            "start": float(run["start"]),
            "end": float(run["last"] + 1),
            "seconds": float(span),
            "read_seconds": len(run["values"]),
            "sign": run["sign"],
            "valence": round(float(np.mean(run["values"])), 3),
            "dominant": max(sorted(counts), key=lambda k: counts[k]),
        })
    episodes.sort(key=lambda e: (-e["seconds"], e["start"]))
    return episodes


def _reliability(coverage_pct: float, mean_confidence: float,
                 mean_run: float, labels: Mapping,
                 dispersion: Mapping) -> dict:
    """Every reason found to distrust the reading, and how much they add up to.

    Assembled from the measurements above rather than asserted, so a caller can
    show the reasons instead of a verdict — which is the difference between a
    user who knows to check the footage and one who is told a number.
    """
    reasons = []
    if coverage_pct < LOW_COVERAGE:
        reasons.append(
            f"only {coverage_pct:.0f}% of the video had a readable face, so "
            f"every share here describes that part of it and not the rest")
    if mean_confidence < LOW_CONFIDENCE:
        reasons.append(
            f"the classifier averaged {mean_confidence:.2f} confidence, which is "
            f"close to picking between near-ties")
    if mean_run and mean_run < UNSTABLE_RUN:
        reasons.append(
            f"the label changed every {mean_run:.1f}s on average, faster than "
            f"anything on screen plausibly does")

    for label, stats in sorted(labels.items()):
        if float(stats.get("share_pct") or 0.0) < DOMINANT_SHARE:
            continue
        spread = dispersion.get(label) or {}
        if spread.get("uniform"):
            reasons.append(
                f"'{label}' is {stats['share_pct']:.0f}% of the reading and is "
                f"spread evenly across the whole file — the shape of a "
                f"classifier that reads this footage one way, rather than of "
                f"anything that happens in it")

    # "unflagged" rather than "good": nothing here inspects whether the labels
    # are *correct*, only whether the run has the shape of one that cannot be.
    # An absence of detected problems is not evidence of accuracy, and a word
    # like "good" would be read as exactly that.
    level = "unflagged"
    if len(reasons) >= 2:
        level = "low"
    elif reasons:
        level = "moderate"
    return {"level": level, "reasons": reasons}


def analyse(expressions: Optional[Mapping],
            duration: float,
            *,
            segments: Optional[Iterable[Sequence[float]]] = None,
            detections: Optional[Mapping] = None,
            buckets: int = DEFAULT_BUCKETS,
            min_episode_seconds: float = 8.0,
            episode_gap: float = 4.0) -> dict:
    """The whole reading of one video: mix, arc, episodes, and what to doubt.

    ``expressions`` is the per-second scan; ``segments`` the kept clips, which
    are given their own reading so a report can say which of them run against
    the file around them. ``detections`` is ``{second: [class, ...]}`` — with it,
    each class gets the reading measured while it was on screen, against the
    file's own baseline.
    """
    labels = _normalise(expressions)
    duration = float(duration or 0.0)
    if not labels or duration <= 0:
        return {}

    seconds = sorted(labels)
    read = len(seconds)
    coverage_pct = round(100.0 * read / duration, 1) if duration else 0.0

    per_label: dict = {}
    for _sec, (label, confidence) in labels.items():
        stats = per_label.setdefault(label, {"seconds": 0, "confidence": []})
        stats["seconds"] += 1
        stats["confidence"].append(confidence)
    for label, stats in per_label.items():
        stats["share_pct"] = round(100.0 * stats["seconds"] / read, 1)
        stats["mean_confidence"] = round(float(np.mean(stats["confidence"])), 3)
        del stats["confidence"]

    values = [valence_of(label, confidence)
              for label, confidence in labels.values()]
    valenced = [v for v in values if v != 0]
    mean_confidence = float(np.mean([c for _l, c in labels.values()]))

    runs = _runs(seconds, labels)
    mean_run = round(float(np.mean([r[3] for r in runs])), 2) if runs else 0.0

    bucket_rows = _buckets(labels, duration, buckets)
    dispersion = {label: _dispersion(bucket_rows, label) for label in per_label}
    dispersion = {k: v for k, v in dispersion.items() if v}

    unvalenced_seconds = sum(per_label.get(label, {}).get("seconds", 0)
                             for label in UNVALENCED)

    out = {
        "coverage": {
            "read_seconds": read,
            "duration": round(duration, 1),
            "pct": coverage_pct,
        },
        "labels": per_label,
        "valence": {
            # Over the seconds that carry a direction at all. Averaging the
            # neutral and unvalenced ones in would pull every file toward zero
            # in proportion to how much of it the model had no opinion about.
            "mean": round(float(np.mean(valenced)), 3) if valenced else 0.0,
            "mean_all_read": round(float(np.mean(values)), 3) if values else 0.0,
            "positive_pct": round(100.0 * sum(1 for v in values if v > 0) / read, 1),
            "negative_pct": round(100.0 * sum(1 for v in values if v < 0) / read, 1),
            "unvalenced_pct": round(100.0 * unvalenced_seconds / read, 1),
        },
        "stability": {
            "runs": len(runs),
            "mean_run_seconds": mean_run,
        },
        "buckets": bucket_rows,
        "arc": _arc(bucket_rows),
        "shift": _shift(bucket_rows),
        "dispersion": dispersion,
        "episodes": _episodes(labels, seconds, min_episode_seconds, episode_gap),
    }
    out["reliability"] = _reliability(coverage_pct, mean_confidence, mean_run,
                                      per_label, dispersion)

    if segments:
        out["segments"] = _segment_readings(labels, segments,
                                            out["valence"]["mean_all_read"])
    if detections:
        out["by_class"] = _class_readings(labels, detections,
                                          out["valence"]["mean_all_read"])
    return out


def _class_readings(labels: Mapping, detections: Mapping,
                    baseline: float) -> list:
    """The expression reading while each detected class is on screen.

    This is the measurement behind the question everyone eventually asks: does
    the reading *differ* while a particular thing is happening? It is answerable
    — a mean over the seconds carrying that class, against the mean over the
    whole file — and it is worth answering precisely because the answer is
    usually no.

    ``delta`` is the entire content of the row. A file whose classifier reads
    negative throughout produces a negative figure for every class in it, and an
    absolute number invites the reader to conclude something from what is only
    the video's baseline restated. A class that matches the baseline is
    reported as matching it, which is a real result and the common one.

    What this cannot do is attribute. Co-occurrence is not cause: the same
    seconds carry camera angle, lighting, shot type and whatever else the
    footage does, and a class that reads differently may be marking any of them.
    The row says the readings differ, never why.
    """
    per_class: dict = {}
    for sec, names in (detections or {}).items():
        found = labels.get(int(sec))
        if not found:
            continue
        for name in names or ():
            per_class.setdefault(str(name), []).append(found)

    rows = []
    for name, found in per_class.items():
        if len(found) < MIN_CLASS_SECONDS:
            continue
        values = [valence_of(label, confidence) for label, confidence in found]
        counts: dict = {}
        for label, _c in found:
            counts[label] = counts.get(label, 0) + 1
        mean = float(np.mean(values))
        delta = mean - baseline
        rows.append({
            "name": name,
            "read_seconds": len(found),
            "valence": round(mean, 3),
            "delta": round(delta, 3),
            "distinguishable": bool(abs(delta) >= CLASS_DELTA),
            "dominant": max(sorted(counts), key=lambda k: counts[k]),
            "shares": {k: round(100.0 * v / len(found), 1)
                       for k, v in counts.items()},
        })
    rows.sort(key=lambda r: (-abs(r["delta"]), r["name"]))
    return rows


def peak_in_range(expressions: Optional[Mapping],
                  start: float,
                  end: float) -> Optional[dict]:
    """The clearest stretch of one label inside a clip, and where it began.

    The report already marks the second a clip was loudest and the second
    movement stopped. This is the third mark of the same kind, and it is the one
    that makes the other two comparable to something a person watches for: a
    reading that arrives *after* the loudest point is a different clip from one
    where it was there all along, and neither the arc nor the per-clip average
    can tell those apart, because both average the clip flat.

    What is returned is the *onset* — the first second of the run, not the
    second the classifier was most sure. Confidence peaks wherever the face
    happened to be most frontal, which is a fact about the camera; the onset is
    the earliest second the reading was there at all, and that is the thing worth
    putting beside another timestamp.

    Two kinds of run are skipped, for the same reason. Neutral ones: timing a
    neutral reading against the loudest point relates a moment to the absence of
    one, which reads as a finding and is not. And the video's own dominant label
    when it holds most of the file — a clip that reads sad inside a file that
    reads sad throughout has told the reader about the file, and putting a
    timestamp on it dresses the baseline up as an event. What is wanted is the
    reading that departs from the norm, which is why the norm has to be measured
    before a clip can be judged against it.

    `surprise` is kept despite carrying no valence — it is unvalenced because
    delight and alarm produce it alike, not because it is uninformative, and it
    is the label most likely to mark the second something happened.

    Returns ``None`` whenever nothing clears the bars, which is common and is
    the point: a clip whose face was unreadable, or whose reading never settled,
    has no second to name and should not be given one.
    """
    labels = _normalise(expressions)
    if not labels:
        return None
    seconds = sorted(labels)
    inside = [s for s in seconds if float(start) <= s < float(end)]
    if not inside:
        return None

    # Runs of one label, contiguous in *read* seconds like `_runs` — a second
    # the face was not visible in should not split a reading in two.
    runs: list = []
    for sec in inside:
        label, confidence = labels[sec]
        if runs and runs[-1]["label"] == label:
            runs[-1]["seconds"].append(sec)
            runs[-1]["confidence"].append(confidence)
        else:
            runs.append({"label": label, "seconds": [sec],
                         "confidence": [confidence]})

    # What this file reads as when nothing in particular is happening. Measured
    # over the whole scan rather than the clip, because a clip is far too short
    # to establish its own norm — and a norm taken from the clip would rule out
    # exactly the run that fills it.
    counts: dict = {}
    for label, _confidence in labels.values():
        counts[label] = counts.get(label, 0) + 1
    dominant = max(sorted(counts), key=lambda k: counts[k])
    pervasive = (100.0 * counts[dominant] / len(labels)) >= DOMINANT_SHARE

    usable = []
    for run in runs:
        if str(run["label"]).lower() == "neutral":
            continue
        if pervasive and run["label"] == dominant:
            continue
        if len(run["seconds"]) < MIN_PEAK_RUN:
            continue
        mean = float(np.mean(run["confidence"]))
        if mean < LOW_CONFIDENCE:
            continue
        usable.append((mean, len(run["seconds"]), run))
    if not usable:
        return None

    # Strongest reading first, then the longest, then the earliest. Mean rather
    # than peak confidence: a run that reads 0.8 throughout is better evidence
    # than one that touched 0.9 on a single frame and 0.4 either side of it.
    mean, length, run = max(usable, key=lambda u: (u[0], u[1], -u[2]["seconds"][0]))
    onset = int(run["seconds"][0])

    # What it turned from, when there is a reading close enough behind it to
    # have turned from anything. Deliberately looks outside the clip too: a
    # change three seconds before the clip starts is still a change.
    previous = [s for s in seconds if s < onset]
    from_label = ""
    if previous:
        last = previous[-1]
        earlier = labels[last][0]
        if onset - last <= TURN_GAP and earlier != run["label"]:
            from_label = earlier

    return {
        "second": onset,
        "timestamp": f"{onset // 60}:{onset % 60:02d}",
        "label": run["label"],
        "confidence": round(mean, 2),
        "seconds": int(length),
        "read_seconds": len(inside),
        "turned": bool(from_label),
        "from_label": from_label,
    }


def _segment_readings(labels: Mapping, segments: Iterable[Sequence[float]],
                      video_mean: float) -> list:
    """Each kept clip's own reading, against the file's.

    ``delta`` is the point: a clip reading -0.4 in a file that reads -0.38 is
    the video's normal, and reporting its absolute figure alone would make every
    clip in a negative-reading file look like a finding.
    """
    rows = []
    for i, span in enumerate(segments, start=1):
        start, end = float(span[0]), float(span[1])
        inside = [(label, confidence)
                  for sec, (label, confidence) in labels.items()
                  if start <= sec < end]
        row = {"index": i, "start": start, "end": end,
               "read_seconds": len(inside)}
        if inside:
            mean = float(np.mean([valence_of(l, c) for l, c in inside]))
            counts: dict = {}
            for label, _c in inside:
                counts[label] = counts.get(label, 0) + 1
            row["valence"] = round(mean, 3)
            row["delta"] = round(mean - video_mean, 3)
            row["dominant"] = max(sorted(counts), key=lambda k: counts[k])
            row["shares"] = {k: round(100.0 * v / len(inside), 1)
                             for k, v in counts.items()}
        rows.append(row)
    return rows
