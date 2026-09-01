"""Run the narration passes over a finished report, in one call.

The chapter walk and the clip read already know how to narrate a report on
disk. What they lacked was somewhere to be called from that is not a menu: both
were reachable only from the AI-summary button, which meant the answer to "what
is in this video" existed only if the user knew to ask for it after the run had
already finished, and then waited through a second pass with the window frozen.

So this module is the joining piece — build the model once, run whichever
passes are turned on, report progress, stop when the run stops. It carries no
Qt and no pipeline state, which is what lets the pipeline call it mid-run and
the window call it from a menu without either growing a copy of the other's
logic.

Both passes are cheap to *decide* and expensive to *run*, so the decision is
made here and stated in the log before the first call: several minutes of
apparent silence is the failure mode this pass has, and the fix is saying what
is about to happen rather than making it faster.

Why the model is built once for both: each pass loading its own would hold two
copies of a vision model in memory at the same moment on a machine that usually
cannot spare one.
"""
from __future__ import annotations

from typing import Callable, Mapping, Optional

# What a run does when nothing says otherwise: both, at both scales. They answer
# different questions and neither substitutes for the other — a clip paragraph
# says what is in these few seconds, a chapter paragraph says what the stretch
# they sit in was doing — so a report with one and not the other can always be
# asked something it cannot answer.
#
# The cost is the honest objection and it is real: this is a model call per
# chapter *plus* one per clip, and on a local model that is the slowest thing
# the report does by a wide margin. It is defaulted on anyway because a run the
# user has already committed minutes to is the cheapest possible moment to spend
# them, and the alternative — discovering afterwards that the question needs a
# second pass over footage now finished — costs the whole wait again. Both
# checkboxes turn it off for a run that does not want it.
DEFAULTS = {"narrate_clips": True, "narrate_chapters": True}


def _entry(config: Mapping) -> dict:
    """The model to narrate with, as the GUI stored it."""
    entry = config.get("narration_model") or {}
    if not isinstance(entry, Mapping):
        return {}
    return dict(entry)


def wanted(config: Mapping) -> tuple:
    """Which passes this run should make: ``(chapters, clips)``."""
    return (bool(config.get("narrate_chapters", DEFAULTS["narrate_chapters"])),
            bool(config.get("narrate_clips", DEFAULTS["narrate_clips"])))


def narrate_report_file(json_path: str,
                        *,
                        config: Mapping,
                        log_fn: Callable[[str], None] = print,
                        cancel_fn: Optional[Callable[[], bool]] = None) -> dict:
    """Run the wanted passes over the report at ``json_path``.

    Returns ``{"chapters": n, "clips": n}`` — how many of each were written.
    Never raises: this runs at the tail of a pipeline that has already produced
    the cut and the report, and a narration that cannot be written is not a
    reason to lose either. Every failure becomes a log line and a zero.
    """
    from modules import advisor
    from modules.llm_models import label_for

    done = {"chapters": 0, "clips": 0}
    do_chapters, do_clips = wanted(config)
    if not (do_chapters or do_clips):
        return done
    if cancel_fn is not None and cancel_fn():
        return done

    entry = _entry(config)
    backend = entry.get("backend") or "ollama"
    name = entry.get("model") or "llama3"
    label = label_for(entry) if entry else f"{backend}/{name}"

    try:
        llm = advisor.load_llm(backend, name, mmproj=entry.get("mmproj"),
                               vision=True)
    except Exception as exc:
        log_fn(f"⚠️ Narration skipped — {label} could not be loaded: {exc}")
        return done
    if llm is None:
        log_fn(f"⚠️ Narration skipped — could not reach {label}. The report "
               "keeps its measurements; only the telling needs a model.")
        return done

    # Asked once, before either pass, because the clip pass is meaningless
    # without it and the chapter pass is merely poorer — see the two passes'
    # own notes. Said as a warning rather than an abort for the chapter walk.
    sees = not hasattr(llm, "accepts_images") or llm.accepts_images()
    if do_clips and not sees:
        log_fn(f"⚠️ {label} has no vision half loaded, so it cannot see the "
               "frames — and the frames are the whole point of the clip pass. "
               "Skipping it.")
        do_clips = False

    if do_chapters:
        done["chapters"] = _pass(
            "chapters", json_path, llm=llm, label=label,
            log_fn=log_fn, cancel_fn=cancel_fn)
    if do_clips and not (cancel_fn is not None and cancel_fn()):
        done["clips"] = _pass(
            "clips", json_path, llm=llm, label=label,
            log_fn=log_fn, cancel_fn=cancel_fn)
    return done


def _pass(which: str, json_path: str, *, llm, label, log_fn, cancel_fn) -> int:
    """One pass, with its failure contained. Returns how many it wrote.

    The two passes are called through one function because the only differences
    between the call sites are the module and the noun, and a second copy of the
    try/except would be a second place for the contract above ("never raises")
    to stop being true.
    """
    if which == "chapters":
        from modules.chapter_story import tell_report_file
        noun = "chapter"
    else:
        from modules.clip_story import tell_report_file
        noun = "clip"

    log_fn(f"📖 Narrating {noun}s with {label} — one call each, so this takes "
           "minutes rather than seconds.")
    try:
        return tell_report_file(json_path, llm=llm, model_name=label,
                                log_fn=log_fn, cancel_fn=cancel_fn) or 0
    except Exception as exc:
        log_fn(f"⚠️ The {noun} narration failed: {exc}")
        return 0
