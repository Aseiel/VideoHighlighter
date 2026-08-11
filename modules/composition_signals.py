"""Gather the per-second signals a composition rule can test.

``CompositionEngine`` takes a mapping of signal name to per-second values and
knows nothing about where any of them came from — which is the point, but it
leaves somebody having to find them. This is that somebody, kept out of
``pipeline.py`` because that file diverges between the two editions and every
line added to it is a line to be ported by hand.

What a rule can refer to
------------------------

``audio_level``
    dBFS per second. Absolute, so a threshold on it is a threshold on the
    mastering as much as on the content; ``vocal_effort`` is almost always the
    one wanted instead.

``vocal_brightness``, ``vocal_onset``, ``vocal_effort``
    From ``modules/vocal_signals.py``, as z-scores against the video's own
    vocal seconds. Zero outside that module's vocal gate — a value describing
    something that is not a voice is not reported as one.

``vocal_density``, ``vocal_density_pct``
    How concentrated the loud seconds are around this one — the measurement
    that actually recovered the hand-labelled episodes (16 of 16 across four
    recordings, one false positive in the region stated to contain none). The
    ``_pct`` form is that value's rank within the file, 0-100, so ``min: 90``
    means "the top tenth of this recording" whatever its mastering. It finds
    *episodes*; it does not judge them — see the caveat below.

``waveform_peak_density``, ``waveform_peak_density_pct``
    How many level peaks fall within 20 seconds of this one.

    **Not the arrows on the AUDIO WAVEFORM row**, despite the name and despite
    what this said before. Those are drawn from 1000 bins for the whole file;
    this is measured at 10 Hz and finds 5-12 times as many peaks. With
    suppression at one second a 20 s window holds at most 20, so ``min: 10``
    means half the seconds carry a peak — sustained level, not spikiness. A
    short burst between quiet stretches scores low, and a steady moderate
    passage scores high with no arrow in sight.

    Measured against 17 spans hand-marked as spontaneous and 5 as performed:
    AUC 0.84 on the raw count. At ``min: 10`` it keeps 15 of the 17 and cuts 4
    of the 5.

    Prefer the raw ``waveform_peak_density`` to the ``_pct`` form here — it
    measured better (0.84 against 0.75), which is the opposite of every other
    signal in this list and is explained in ``vocal_signals.peak_density_curve``.
    Read that before relying on it: the advantage comes from a file-level
    difference in peak rate that may be production style rather than
    performance. And five negatives is few enough that only the first digit of
    "4 of 5" means anything. A filter, not a verdict.

``expression``
    The expression reading for the second, as a label. Only the clearest face
    in each second, which is ``face_scan``'s reduction, not one made here.

Nothing in this list is specific to any subject matter, and none of it decides
what a combination means. A rule names the signals, sets the thresholds and
supplies the event name; all three are the user's.

**What these do not measure.** Density finds sustained, concentrated vocal
episodes, and it finds them whether or not they are spontaneous. Tested against
labels on a recording described as containing one genuine episode and otherwise
performance, it returned twenty — which is the right answer to "where are the
episodes" and the wrong answer to any question about authenticity. Nothing here
distinguishes the two, and a rule built on these signals should not be named or
described as though it does.

Cost
----

The vocal measurement decodes the audio and runs an FFT per second — tens of
seconds on a feature-length file, which is fine once and much too slow for a
loop where the point is to edit a rule and see the result. It is therefore
cached beside the analysis cache and keyed by the video's size and mtime, so
re-applying rules pays for it once.
"""
from __future__ import annotations

import json
import os
from typing import Callable, Mapping, Optional

# Bumped when the measurement changes in a way that makes older files wrong.
# Without it a cached curve from a previous definition is silently reused and
# the rules are evaluated against numbers that no longer mean what they say.
CACHE_VERSION = 3


def _video_key(video_path: str) -> str:
    try:
        st = os.stat(video_path)
        return f"{st.st_size}:{int(st.st_mtime)}:{CACHE_VERSION}"
    except OSError:
        return f"missing:{CACHE_VERSION}"


def _cache_path(video_path: str, cache_dir: str) -> str:
    import hashlib

    digest = hashlib.sha256(
        os.path.abspath(video_path).encode("utf-8", "replace")).hexdigest()[:16]
    return os.path.join(cache_dir, f"{digest}.vocal.json")


def vocal_curves(video_path: str,
                 *,
                 cache_dir: str = "./cache",
                 progress: Optional[Callable] = None,
                 cancel=None,
                 log_fn=print) -> dict:
    """``vocal_signals.analyse`` with a disk cache. ``{}`` when it cannot run.

    Returns empty rather than raising: a rule set may be entirely spatial, and
    losing a run because an audio measurement failed would be a bad trade for a
    signal nothing was asking about.
    """
    path = _cache_path(video_path, cache_dir)
    key = _video_key(video_path)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            cached = json.load(fh)
        if cached.get("key") == key:
            return cached.get("curves") or {}
    except (OSError, ValueError):
        pass

    try:
        from modules import vocal_signals
        result = vocal_signals.analyse(video_path, progress=progress,
                                       cancel=cancel)
    except Exception as exc:
        log_fn(f"⚠️ Vocal signals unavailable: {exc}")
        return {}

    curves = result.get("curves") or {}
    if not result.get("enough_voice"):
        # The z-scores are computed against the video's own vocal seconds; too
        # few of them and they describe a handful of samples rather than a
        # voice. Saying so beats letting a rule fire on a normalisation nobody
        # would trust.
        log_fn(f"ℹ️ Vocal signals: only {result.get('vocal_seconds', 0)} vocal "
               "second(s) — brightness and onset are not normalised, so "
               "rules using them will not fire.")
    try:
        os.makedirs(cache_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"key": key, "curves": curves}, fh)
    except OSError:
        pass
    return curves


def expression_labels(video_path: str, *, cache_dir: str = "./cache") -> dict:
    """``{second: label}`` from an existing expression scan, or ``{}``.

    Read-only: this never starts a scan. A scan needs a model and a decode of
    the whole video, and starting one as a side effect of applying a rule would
    turn a millisecond operation into a long one without being asked.
    """
    try:
        from modules.face_scan import best_by_second, cache_path_for, load
        scanned = load(cache_path_for(video_path, cache_dir))
        if not scanned:
            return {}
        return {int(sec): str(label)
                for sec, (label, _confidence) in best_by_second(scanned).items()
                if label}
    except Exception:
        return {}


def gather(video_path: str,
           *,
           cache_dir: str = "./cache",
           levels: Optional[Mapping] = None,
           needed: Optional[set] = None,
           progress: Optional[Callable] = None,
           cancel=None,
           log_fn=print) -> dict:
    """Every signal a rule set could ask for, ready for ``CompositionEngine.run``.

    ``needed`` is the set of signal names the current rules actually mention.
    Passing it skips the measurements nothing refers to, which matters because
    the vocal one is the expensive member of this list and most rule sets are
    spatial. With ``needed`` omitted, everything is gathered.
    """
    wanted = None if needed is None else {str(n) for n in needed}

    def asked(*names) -> bool:
        return wanted is None or any(n in wanted for n in names)

    signals: dict = {}

    if levels is not None and asked("audio_level"):
        signals["audio_level"] = list(levels)

    if asked("vocal_effort", "vocal_brightness", "vocal_onset",
             "vocal_density", "vocal_density_pct",
             "waveform_peak_density", "waveform_peak_density_pct"):
        curves = vocal_curves(video_path, cache_dir=cache_dir,
                              progress=progress, cancel=cancel, log_fn=log_fn)
        if curves:
            signals["vocal_effort"] = curves.get("effort") or []
            signals["vocal_brightness"] = curves.get("brightness_z") or []
            signals["vocal_onset"] = curves.get("onset_z") or []
            signals["vocal_density"] = curves.get("density") or []
            signals["vocal_density_pct"] = curves.get("density_pct") or []
            signals["waveform_peak_density"] = curves.get("peak_density") or []
            signals["waveform_peak_density_pct"] = (
                curves.get("peak_density_pct") or [])
            signals.setdefault("audio_level", curves.get("level") or [])

    if asked("expression"):
        labels = expression_labels(video_path, cache_dir=cache_dir)
        if labels:
            signals["expression"] = labels

    return signals


def signal_names(rules_path: Optional[str]) -> set:
    """Which signals a rules file mentions, without building the engine.

    Lets a caller gather only what is referred to. Parsed rather than inferred
    from a loaded engine so a malformed rules file costs nothing here — the
    engine reports that failure, and reporting it twice would be noise.

    Unticked events are skipped: the vocal measurement decodes the whole file,
    and paying for it because of a rule that is switched off is exactly the
    wait this list exists to avoid.
    """
    if not rules_path or not os.path.exists(rules_path):
        return set()
    try:
        import yaml

        with open(rules_path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        return {
            str(condition["signal"])
            for event in (raw.get("events") or [])
            if event.get("enabled", True)
            for condition in (event.get("signals") or [])
            if condition.get("signal")
        }
    except Exception:
        return set()
