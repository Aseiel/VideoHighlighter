"""Decide what a *second* run still has to compute, given what the cache holds.

This module exists because one bug kept coming back in different clothes.

The analysis cache keys on analysis *inputs* — what to look for, at what
sampling rate, with which detector. Scoring **points** are deliberately not part
of that key, and rightly so: changing a weight must not discard an hour of
transcription (see ``modules.video_cache.build_analysis_cache_params``).

But several points do double duty. They are also the switch that decides whether
a detector runs at all — "all motion points are zero, so don't bother detecting
motion". The moment a setting both (a) gates an analysis artifact and (b) sits
outside the cache signature, this happens:

1. A run with the points at zero skips the detector and caches an empty list.
2. The user raises the points, hoping for more signal.
3. Same inputs, so the same cache key: the cached pass loads the empty list.
4. Nothing re-detects. The new weight scores against nothing, forever.

The failure is silent because **an empty list is a legal answer**. "Detected
nothing" and "never looked" are stored identically, and no later code can tell
them apart. Every instance so far — motion, audio peaks, the report's detection
boxes, composed event names, the waveform — was found by a user noticing a
feature had quietly stopped working, then fixed by hand at that one call site.

So the predicate gets a name, one implementation, and a test that fails when a
new gated artifact is added without handling this. The point is not that
:func:`needs_backfill` is hard — it is three clauses — but that it is *declared*
in :data:`GATED_ARTIFACTS`, where the invariant test can see it.

Nothing here does I/O or knows what a detector is. It answers one question about
two dicts, which is what makes the second-run behaviour testable without a video
— see ``tests/test_analysis_plan.py``.
"""
from __future__ import annotations

from typing import Iterable, Mapping, Sequence

# Analysis artifacts whose computation is gated by a *scoring* setting rather
# than by the cache signature. Each entry maps a name to the settings that turn
# it on: if every one of them is falsy, the pipeline skips the detector and
# caches nothing, which is the situation this module exists to notice.
#
# Adding a detector that is skipped when a weight is zero? Add it here. The
# invariant test in tests/test_analysis_plan.py fails otherwise, which is the
# whole point — it is the part that stops this bug coming back a fourth time.
GATED_ARTIFACTS: dict[str, tuple[str, ...]] = {
    # scenes + motion_events + motion_peaks all come from one detector pass.
    "motion": ("scene_points", "motion_event_points", "motion_peak_points"),
    "audio_peaks": ("audio_peak_points",),
}

# Settings that gate an artifact *and* are part of the cache signature, so
# changing one already forces a full re-analysis and no backfill is needed.
# Listed rather than inferred so the invariant test can tell "handled by the
# signature" apart from "nobody thought about it".
SIGNATURE_GATED: dict[str, tuple[str, ...]] = {
    "objects": ("highlight_objects",),
    "actions": ("interesting_actions",),
    "transcript": ("use_transcript", "search_keywords"),
}


def gate_is_open(gui_config: Mapping, artifact: str) -> bool:
    """True when this run's settings ask for ``artifact`` to exist at all.

    Any one setting being truthy is enough: the motion detector produces scenes,
    events and peaks in a single pass, so wanting any of the three means the
    pass has to happen.
    """
    settings = GATED_ARTIFACTS.get(artifact, ())
    return any(bool(gui_config.get(name)) for name in settings)


def is_empty(*values: Iterable) -> bool:
    """True when every artifact value is absent or empty.

    The reason the caller cannot simply write ``not scenes``: an artifact is
    usually several lists that come from one detector pass, and any one of them
    being populated proves the pass ran. Requiring *all* of them to be empty is
    what keeps a genuinely silent stretch of video from being mistaken for a
    detector that never ran.
    """
    return not any(bool(v) for v in values)


def needs_backfill(artifact: str, gui_config: Mapping, *,
                   using_cache: bool, values: Sequence) -> bool:
    """Must ``artifact`` be computed even though this is a cached run?

    Only when all three hold: the run is reusing a cache, this run's settings
    want the artifact, and the cache has nothing for it. A cached run whose
    artifact is populated is left alone — that is the cache doing its job — and
    a run that does not want the artifact does not compute it just because the
    cache lacks one.
    """
    if not using_cache:
        return False            # a fresh run computes what it wants anyway
    if not gate_is_open(gui_config, artifact):
        return False            # nothing asked for it
    return is_empty(*values)


def plan(gui_config: Mapping, *, using_cache: bool,
         values: Mapping[str, Sequence]) -> dict[str, bool]:
    """``{artifact: must_compute}`` for every gated artifact, in one call.

    ``values`` maps an artifact name to whatever the cache produced for it.
    Artifacts absent from it are treated as empty, so a caller that has not
    loaded one yet gets the safe answer rather than a KeyError.
    """
    return {
        name: needs_backfill(name, gui_config, using_cache=using_cache,
                             values=values.get(name) or ())
        for name in GATED_ARTIFACTS
    }


def describe(artifact: str) -> str:
    """The log line for a backfill, so every call site words it the same way."""
    return (f"🔁 Cache has no {artifact.replace('_', ' ')} data but this run "
            "scores it — computing now (the rest of the cached analysis is kept).")
