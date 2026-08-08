"""What it would take to measure something this run had no signal for.

:mod:`modules.vocabulary_gap` and :mod:`modules.uncovered_claims` both end in
the same place: *this was said, and nothing here was watching for it*. The
question a user asks next is not "which weight do I change" — no weight helps —
but **"so how would I detect it, and is it worth the trouble?"** That question
has a small number of real answers in this application, they differ by an order
of magnitude in cost, and choosing badly between them is expensive: teaching a
category takes minutes and training a class takes an afternoon, and people
reliably reach for the second when the first would have done.

So the answers are enumerated here, with what each one *gives* you and what it
*costs*, and two of them are picked: the cheapest thing that would actually
measure it, and the most reliable thing available. Two rather than a list,
because a list of six routes is a decision handed back to the person who asked.

What this module does not know
------------------------------

**What was said.** It is never given the claim, and could not use it — deciding
that a spoken word describes an object rather than a movement is a judgement
about subject matter, and this repo ships no vocabulary to make it with (see
CLAUDE.md, and :mod:`modules.vocabulary_gap` for the same refusal). So a route
is not selected by what the thing *is*. Each one instead carries the condition
it holds under, in the user's own terms — "if you can point at frames showing
it", "if it fills a good part of the frame" — and the reader, who can see their
own footage, settles in a second what no lexicon here could settle at all.

**Whether it will work.** Every figure below is a cost, not a promise. The one
route whose success is genuinely unpredictable — an open-vocabulary query, which
is excellent on ordinary things and can be nearly blind on specialised subject
matter — is therefore offered as a *probe* rather than as a recommendation: five
minutes with a control query says whether it can see your subject at all, and
that answer is worth having before committing to an afternoon of labelling.

What it does know is which routes this particular run has the prerequisites for,
and that is read from the record: how many classes the detector produced (a
composition rule needs at least two to arrange), whether a CLIP index was ever
built (chapters say so — their `method` is "visual" when one was), whether a
face scan ran. A route offered without its prerequisite is a route the user
discovers is unavailable after deciding on it.

And which engines this *build* has at all. Each route names the module it needs
and is dropped when that module is not importable, which is what lets one file
serve both editions honestly: the editions ship different engines, and a route
recommending something this build cannot run is the same failure as one
recommending a class the detector never emits — it costs the user a decision
before they find out. Detected rather than declared, so no edition flag has to
be threaded here and nothing goes stale when the boundary moves.

The costs quoted here come from ``docs/DETECTION-GUIDE.md`` and are measured,
not estimated. When they change, change them there and here together.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Mapping, Optional, Sequence

# Effort and confidence are ordinals, not scores. They exist to be *compared* —
# "cheaper than", "more reliable than" — and any arithmetic on them would be
# inventing a precision that measuring a route's cost in minutes does not have.
EFFORT_INSTANT, EFFORT_MINUTES, EFFORT_HOURS, EFFORT_SESSION = 0, 1, 2, 3
CONFIDENCE_PROXY, CONFIDENCE_UNEVEN, CONFIDENCE_GOOD, CONFIDENCE_EXACT = 0, 1, 2, 3


@dataclass
class Route:
    """One way to get a signal the run does not have.

    ``holds_when`` and ``fails_when`` are the load-bearing fields. The cost of a
    route is a fact and the same for everybody; whether it applies depends on
    what the user is looking for, which only they can see, so every route states
    its condition instead of pretending the choice was made for them.
    """
    id: str
    name: str
    gives: str                 # what you get out of it
    effort: str                # what it costs, in time and what it needs
    effort_rank: int
    confidence: str            # how much the answer is worth
    confidence_rank: int
    holds_when: str            # when this is the right route
    fails_when: str            # when it is not, said before they spend the time
    repeat: str                # cost of asking a second question
    topic: str                 # page in docs/advisor that explains it
    needs: str = ""            # prerequisite in this run, "" when always usable
    module: str = ""           # engine this build must have, "" when always
    # The cheap test this route can be turned into, when it has one: not a way
    # to measure the thing, a way to find out in five minutes whether the
    # expensive route is unavoidable.
    probe: Optional[dict] = None

    def as_dict(self) -> dict:
        return asdict(self)


# Ordered as a person would work through them: cheapest first. `pick` does not
# rely on this order — it sorts — but a reader of `all` gets the sequence that
# the recommendation is drawn from rather than an arbitrary one.
ROUTES: tuple = (
    Route(
        id="compose",
        name="Compose it from classes this video already produced",
        gives=("a label per second, exact counts, and the one thing scores "
               "cannot express — that something is not there"),
        effort=("minutes to write, and one re-run of object detection: a "
                "cached detection pass skips the composition engine, so a new "
                "rule cannot fire until detection runs again"),
        effort_rank=EFFORT_MINUTES,
        confidence=("exact where it applies — counting boxes is arithmetic, "
                    "not similarity, so there is no threshold to tune"),
        confidence_rank=CONFIDENCE_EXACT,
        holds_when=("what was said is an arrangement of things this detector "
                    "already finds — one inside another, several at once, none "
                    "of them present"),
        fails_when=("what was said is a thing in its own right that no class "
                    "covers. No rule can conjure a class; it can only arrange "
                    "the ones that exist"),
        repeat="instant — rules read detections that are already cached",
        topic="composition",
        needs="at least two classes detected in this video",
    ),
    Route(
        id="clip_search",
        name="Search the video for the words that were said",
        gives=("a score per sampled frame — how much it looks like the phrase "
               "you typed. No box, nothing to count"),
        effort=("minutes, and the words are already in the transcript. One "
                "pass embeds the video; after that a query is arithmetic on a "
                "small array and costs nothing"),
        effort_rank=EFFORT_MINUTES,
        confidence=("uneven, and for a reason worth knowing: it matches the "
                    "wording. A phrase the model has seen described that way "
                    "works well; one it has not is a low score about nothing"),
        confidence_rank=CONFIDENCE_UNEVEN,
        holds_when=("the thing can be put into plain words — and something "
                    "said out loud on camera usually can, which is what makes "
                    "this the first thing to try on a spoken claim"),
        fails_when=("it is small in the frame, or the wording is unusual. The "
                    "embedding describes the whole picture, and no threshold "
                    "recovers a subject the model has no representation for"),
        repeat="free — the index is built once and answers any number of queries",
        topic="training",
        module="llm.clip_index",
        probe={
            "why": ("Before spending a session on labels, spend five minutes "
                    "finding out whether you have to. Search the video for the "
                    "words that were said, and for a control phrase describing "
                    "something ordinary you know is in the shot. If the "
                    "control finds its moments and your phrase does not, no "
                    "threshold will fix it and a trained class is the answer. "
                    "If both find theirs, you have just avoided the session."),
            "how": ('python -m llm.clip_index --video "your.mp4" --interval 2 '
                    '--query "your thing" --query "a close-up" --topk 10'),
        },
    ),
    Route(
        id="example_category",
        name="Teach a category from example frames",
        gives=("a score per sampled second — how much this moment looks like "
               "the frames you pointed at. No box, so nothing to count"),
        effort=("minutes. Point at a few frames and name them; no dataset, no "
                "labels, no GPU. The first search over a video pays for one "
                "embedding pass, and every later query is free"),
        effort_rank=EFFORT_MINUTES,
        confidence=("good for anything that fills a decent part of the frame, "
                    "and the score is calibrated — a low number means the "
                    "match is weak, not that the scale is off"),
        confidence_rank=CONFIDENCE_GOOD,
        holds_when=("you can point at frames that show it. This is the only "
                    "route that works when you can recognise something on "
                    "sight but cannot put it into words"),
        fails_when=("it is small in the frame. The embedding describes the "
                    "scene, so a few percent of the picture is drowned by "
                    "everything around it — that case needs a detector"),
        repeat="free — the index is built once and answers any number of queries",
        topic="training",
        module="llm.clip_categories",
    ),
    Route(
        id="open_vocabulary",
        name="Type the word into an open-vocabulary detector",
        gives="real boxes and real counts, with no training at all",
        effort=("minutes to try on a short window; around 3 seconds per "
                "frame on CPU, so a whole video is hours rather than minutes"),
        effort_rank=EFFORT_HOURS,
        confidence=("uneven, and measurably so. Excellent on everyday things; "
                    "on specialised subject matter it can be close to blind, "
                    "and no threshold rescues that"),
        confidence_rank=CONFIDENCE_UNEVEN,
        holds_when=("the thing is an ordinary object with edges, and you can "
                    "name it in plain words"),
        fails_when=("it is specialised, or it is an event rather than a thing. "
                    "This detector finds objects with edges, and an event has "
                    "none"),
        repeat="a full re-run per query set",
        topic="training",
        module="llm.owl_detect",
        probe={
            "why": ("Before spending a session on labels, spend five minutes "
                    "finding out whether you have to. Run the open-vocabulary "
                    "detector over a short window with your query and a "
                    "control query for something ordinary you know is in the "
                    "shot. If the control scores well and yours does not, no "
                    "threshold will save it and the trained class is the "
                    "answer. If both score, you have just avoided the "
                    "session."),
            "how": ('python -m llm.owl_detect --video "your.mp4" '
                    '--query "your thing" --query sofa --interval 10 '
                    '--start 600 --end 720'),
        },
    ),
    Route(
        id="action_model",
        name="Use an action model",
        gives="a label over a window of time rather than a single frame",
        effort=("nothing to set up for the 400 everyday actions it already "
                "knows; a session to train a class of your own from folders of "
                "example clips"),
        effort_rank=EFFORT_HOURS,
        confidence=("good for anything defined by how it moves — this is the "
                    "only engine here that sees time at all"),
        confidence_rank=CONFIDENCE_GOOD,
        holds_when=("what was said is a movement — something that only "
                    "exists across several seconds and is invisible in any one "
                    "frame"),
        fails_when=("two things you want to tell apart differ only by where "
                    "they happen. The model is fed a cropped region, so that "
                    "difference is gone before it votes — train one class and "
                    "split it with a composition rule instead"),
        repeat="a full re-run",
        topic="training",
    ),
    Route(
        id="face_category",
        name="Teach a face category from example crops",
        gives="a score per face, for whatever it is about the face you mean",
        effort=("minutes. Pick a handful of face crops; the faces themselves "
                "are already found by the scan this run did"),
        effort_rank=EFFORT_MINUTES,
        confidence=("good, and unlike the built-in expression classes it is "
                    "not limited to the seven a classifier was trained on"),
        confidence_rank=CONFIDENCE_GOOD,
        holds_when=("what was said is about a face — where someone is looking, "
                    "how they are lit, what they are doing with their features"),
        fails_when=("nothing about it is on a face, or the faces are too small "
                    "or too turned away for the scan to have found them"),
        repeat="free once the video has been scanned",
        topic="training",
        needs="a face scan in this run",
        module="modules.face_examples",
    ),
    Route(
        id="trained_class",
        name="Train a class of your own",
        gives=("boxes at frame rate: countable, usable in composition rules, "
               "usable live, and reusable on every video you analyse after"),
        effort=("a session, and a GPU. Collect frames from the conditions it "
                "fails in, label them, train, export. A third of the set "
                "should be frames containing whatever it will confuse for the "
                "target, with nothing boxed"),
        effort_rank=EFFORT_SESSION,
        confidence=("the highest available here, and the only one that stays "
                    "reliable enough to drive a rule"),
        confidence_rank=CONFIDENCE_EXACT,
        holds_when=("it matters enough to justify the labelling, or nothing "
                    "cheaper could see it"),
        fails_when=("your labels do not contain the conditions it fails in. "
                    "Ten frames of the case that breaks it beat a thousand "
                    "more of what already works"),
        repeat="none — a trained class runs with every future analysis",
        topic="training",
    ),
    Route(
        id="spoken_marker",
        name="Score the moments where it is talked about",
        gives=("the seconds the transcript says it, which is not the same "
               "thing and must not be reported as if it were"),
        effort="instant — a transcript keyword weight and a re-score, no re-analysis",
        effort_rank=EFFORT_INSTANT,
        confidence=("a proxy, and a weak one. It measures the talking. A thing "
                    "can be discussed long after it happened, or happen with "
                    "nobody saying a word"),
        confidence_rank=CONFIDENCE_PROXY,
        holds_when=("you want the moments about it now, while deciding "
                    "whether one of the routes above is worth the time"),
        fails_when="you need to know when the thing itself was on screen",
        repeat="free",
        topic="weights",
        needs="a transcript",
    ),
)

BY_ID = {route.id: route for route in ROUTES}


def capabilities(report: Mapping) -> dict:
    """What this run has, as the prerequisites the routes are stated in.

    Read from the record rather than from configuration, for the reason
    :func:`modules.vocabulary_gap.observed_classes` gives: what a detector
    *could* emit and what it emitted in this file are different lists, and only
    the second one supports a recommendation.
    """
    settings = report.get("settings") or {}
    activity = settings.get("detector_activity") or {}
    vocabulary = report.get("vocabulary") or {}
    chapters = report.get("chapters") or []
    return {
        "classes": [str(c) for c in (vocabulary.get("classes") or [])],
        "events": [str(e) for e in (vocabulary.get("events") or [])],
        # A CLIP index is what makes the example route instant rather than a
        # pass over the video, and the chapters record whether one existed:
        # they are cut on it when it is there and fall back to shot length
        # when it is not.
        "clip_index": any(str(ch.get("method") or "") == "visual"
                          for ch in chapters),
        "faces": int(activity.get("face") or 0) > 0,
        "actions": int(activity.get("action") or 0) > 0,
        "transcript": bool(report.get("speech")),
        "engines": [route.id for route in ROUTES if _installed(route.module)],
    }


# {module path: importable}. Answered once per process -- the answer cannot
# change while the app is running, and `find_spec` walks the path every call.
_present: dict = {}


def _installed(module: str) -> bool:
    """Whether this build actually ships the engine a route needs.

    Detected, not declared. The two editions ship different engines and the
    boundary between them moves; an edition flag threaded through here would be
    one more thing to remember to update, and the failure when it went stale
    would be a recommendation the user cannot act on.
    """
    if not module:
        return True
    if module not in _present:
        try:
            from importlib.util import find_spec
            _present[module] = find_spec(module) is not None
        except (ImportError, ValueError):           # pragma: no cover - defensive
            _present[module] = False
    return _present[module]


def available(caps: Mapping, installed: Optional[Callable] = None) -> list:
    """The routes this run can actually offer, cheapest first.

    ``installed`` decides whether a build has a given engine; it defaults to a
    real import check and is passed in by tests, which have to be able to
    describe a build other than the one they are running on — the same file
    ships in both editions and has to be right in each.
    """
    installed = installed or _installed
    classes = list(caps.get("classes") or []) + list(caps.get("events") or [])
    out = []
    for route in ROUTES:
        if not installed(route.module):
            continue
        if route.id == "compose" and len(set(classes)) < 2:
            continue
        if route.id == "face_category" and not caps.get("faces"):
            continue
        if route.id == "spoken_marker" and not caps.get("transcript"):
            continue
        out.append(route)
    return out


def pick(report: Mapping, installed: Optional[Callable] = None) -> dict:
    """The two routes worth naming, plus the free checks that come before them.

    *Fastest* is the least effort that would genuinely measure the thing, and
    *strongest* is the most reliable answer available. They are occasionally the
    same route, and when they are, one is returned rather than one pretending
    to be two — an advisor that always lists exactly two options is one that has
    padded to a number.

    Two routes are deliberately not in that contest.

    **Composing it from existing classes** is not a peer of the others, because
    whether it is possible at all is not a matter of effort: either what was
    said is an arrangement of classes this video already produced, or no rule
    can express it. That question is already answered for free elsewhere —
    :mod:`modules.rule_proposal` asks a model for the rule and reports back that
    it cannot be built — so this is returned as the thing to try *first*, before
    spending anything.

    **The proxy route** is never picked as either. It measures the talking, and
    offering it as "the fast way to measure this" would be the specific
    dishonesty this whole section exists to prevent; it is returned separately,
    labelled as the stopgap it is.
    """
    caps = capabilities(report)
    routes = available(caps, installed)
    ordering = {route.id: index for index, route in enumerate(ROUTES)}
    measuring = [r for r in routes
                 if r.confidence_rank > CONFIDENCE_PROXY and r.id != "compose"]
    out: dict = {"capabilities": caps,
                 "all": [r.as_dict() for r in routes]}
    if BY_ID["compose"] in routes:
        out["first"] = BY_ID["compose"].as_dict()
    if not measuring:
        return out

    # Ties are broken by the catalogue's own order, which is the sequence a
    # person would work through — not by whichever route the dataclass happened
    # to sort next to.
    fastest = min(measuring,
                  key=lambda r: (r.effort_rank, -r.confidence_rank,
                                 ordering[r.id]))
    strongest = max(measuring,
                    key=lambda r: (r.confidence_rank, -r.effort_rank,
                                   -ordering[r.id]))
    out["fastest"] = fastest.as_dict()
    if strongest.id != fastest.id:
        out["strongest"] = strongest.as_dict()

    # The control test, and it is only worth naming when the answer it gives
    # would change what the user does. If the strongest route on offer already
    # costs minutes, spending five of them finding out whether an even cheaper
    # one might work saves nothing.
    #
    # Which test, of the ones this build has, is the *last* route carrying one:
    # the catalogue runs cheapest to dearest, and the dearest zero-training
    # engine is the one whose silence best predicts that training is
    # unavoidable. A detector that cannot see the thing settles the question in
    # a way a whole-frame similarity score cannot.
    probing = [r for r in routes if r.probe]
    if strongest.effort_rank >= EFFORT_SESSION and probing:
        chosen = probing[-1]
        out["probe"] = dict(chosen.probe, route=chosen.id)
        # On a build where the cheapest route is also the best test, the probe
        # is not a separate errand — it is how you find out whether the route
        # you were going to take anyway is working. Saying "first, do this"
        # about the thing already recommended two lines down reads as a page
        # that has lost track of itself.
        if chosen.id == fastest.id:
            out["probe"]["same_as_fastest"] = True

    interim = next((r for r in routes if r.confidence_rank == CONFIDENCE_PROXY),
                   None)
    if interim is not None:
        out["interim"] = interim.as_dict()
    return out


def describe(picked: Mapping) -> list:
    """The picks as lines a page or a prompt can print, in reading order.

    One rendering, used by both, so the page and the narration cannot end up
    recommending different routes for the same run.
    """
    lines = []
    first = picked.get("first")
    if first:
        lines.append(f"Costs nothing to rule out: {first['name']} — "
                     f"{first['holds_when']}. Ask the advisor to draft the "
                     f"rule; it says so when the claim cannot be built from "
                     f"the classes this video has, and that answer is free.")
    probe = picked.get("probe")
    if probe:
        lines.append(("How to tell in five minutes whether it is working: "
                      if probe.get("same_as_fastest")
                      else "Then, before committing: ") + probe["why"])
    for key, lead in (("fastest", "Fastest"), ("strongest", "Most reliable"),
                      ("interim", "Meanwhile")):
        route = picked.get(key)
        if not route:
            continue
        lines.append(f"{lead}: {route['name']} — {route['effort']}. "
                     f"Gives you {route['gives']}. "
                     f"Right route when {route['holds_when']}; "
                     f"not when {route['fails_when']}.")
    return lines
