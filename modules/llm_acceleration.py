"""Which backend narration should use on this machine, and why.

Narration is the most expensive thing a run does, and on Intel hardware the
difference between the two available paths is large: measured on an Arc A750,
41 tok/s against 8.85 with a picture in the context, and an image encode of
about a second against twelve. The second figure is the structural one — a
vision tower is an OpenVINO graph and runs on the GPU, where llama.cpp falls
back to the CPU for it whichever GPU backend it was given. `docs/INTEL-GPU.md`
has the measurements.

So on Intel the faster path should be the one taken, and the user should not
have to know any of the above to get it.

The reason this is a module rather than an `if` is that **the fast path can be
unavailable for four different reasons**, and each needs a different sentence.
No Intel GPU is not a problem at all. No `openvino-genai` is a missing install.
No converted model is a setup step nobody has done yet — the one thing here
that costs an hour and several gigabytes, and the thing a user is most likely
to have skipped without realising it was expected. Silence, or one generic
"falling back to Ollama", would leave all four looking identical.

Nothing here loads a model or imports a heavy library at module scope: this is
asked during preflight, before a run commits to anything.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Optional

# Where a converted model is looked for when the config does not name one. Not
# a download location and not created here — it is only the first place worth
# checking, so that someone who converted a model into the obvious directory
# does not also have to configure it.
DEFAULT_MODEL_DIRS = (
    os.path.join(os.path.expanduser("~"), "ov-models"),
    os.path.join("D:", os.sep, "ov-models"),
)

# A converted model is a directory, not a file, and this is the part of it that
# is always present and never present in a GGUF folder or an Ollama blob store.
IR_MARKER = "openvino_language_model.xml"


def _is_ir_dir(path: str) -> bool:
    return bool(path) and os.path.isfile(os.path.join(path, IR_MARKER))


def converted_models(extra_dirs=()) -> list:
    """Every converted model this machine already has, newest name order.

    Cheap: a directory listing and one `isfile` per candidate.
    """
    found = []
    for root in (*extra_dirs, *DEFAULT_MODEL_DIRS):
        if not root or not os.path.isdir(root):
            continue
        try:
            entries = sorted(os.listdir(root))
        except OSError:
            continue
        for name in entries:
            path = os.path.join(root, name)
            if _is_ir_dir(path) and path not in found:
                found.append(path)
    return found


def intel_gpu() -> Optional[str]:
    """The name of an Intel GPU OpenVINO can use, or None.

    Returns the name rather than a bool so the log line can say which card it
    found — on a machine with more than one that is the difference between a
    useful message and a confusing one.
    """
    try:
        import openvino as ov  # lazy: heavy, and absent on some installs
    except Exception:
        return None
    try:
        core = ov.Core()
        if "GPU" not in core.available_devices:
            return None
        name = str(core.get_property("GPU", "FULL_DEVICE_NAME"))
    except Exception:
        return None
    return name if "intel" in name.lower() else None


def genai_available() -> bool:
    try:
        import openvino_genai  # noqa: F401
        return True
    except Exception:
        return False


def same_model(tag: str, model_dir: str) -> bool:
    """Whether a converted directory holds the model an Ollama tag names.

    `qwen3-vl:8b` and `Qwen3-VL-8B-Thinking-int4-ov` are the same weights in two
    formats; `qwen2.5vl:7b` and that directory are not. The distinction matters
    because accelerating a model is a change of engine the user does not need to
    approve, and swapping it for a different one is a change of answer, which
    they do.

    Punctuation is stripped from both sides because the two naming conventions
    disagree about it and about nothing else.
    """
    keep = "".join(c for c in (tag or "").lower() if c.isalnum())
    have = "".join(c for c in os.path.basename(model_dir or "").lower()
                   if c.isalnum())
    return bool(keep) and keep in have


def dir_reasons(model_dir: str) -> bool:
    """Whether a converted model reasons before answering.

    Read off the chat template, which is where the think tags live, and is the
    same signal `_OpenVINOBackend` uses at load.
    """
    try:
        with open(os.path.join(model_dir, "chat_template.jinja"),
                  encoding="utf-8") as fh:
            return "</think>" in fh.read()
    except OSError:
        return False


def tag_reasons(tag: str, base_url: str = "http://localhost:11434"):
    """Whether an Ollama tag reasons, or None when the server cannot say.

    Best-effort and never fatal: this runs at preflight, and a missing server is
    already reported elsewhere with a better message than this could give.
    """
    try:
        import requests
        resp = requests.post(f"{base_url}/api/show", json={"model": tag},
                             timeout=5)
        if not resp.ok:
            return None
        return "thinking" in (resp.json().get("capabilities") or [])
    except Exception:
        return None


def decide(config: Mapping, entry: Mapping | None = None) -> dict:
    """Pick the narration backend, and say why in a sentence a user can act on.

    Returns ``{"backend", "model", "reason", "accelerated"}``. ``reason`` is
    always set and always worth printing: when the fast path was taken it says
    so, and when it was not it names the one thing standing in the way.

    **An Ollama tag is not a decision against OpenVINO.** The model picker
    stores `backend: ollama` for every Ollama tag, so treating that as an
    explicit choice would mean nobody was ever accelerated - which is what
    happened when this function first shipped. A tag is a choice of *model*,
    and running the same model on the faster engine keeps the choice.

    A GGUF path is different and is left alone: someone who went and found a
    file meant that file.
    """
    entry = dict(entry or {})
    chosen = (entry.get("backend") or "").strip().lower()
    tag = entry.get("model") or ""

    if chosen == "llama-cpp":
        return {"backend": chosen, "model": tag, "accelerated": False,
                "reason": "llama-cpp with the GGUF that was chosen."}

    if chosen == "openvino":
        path = entry.get("model_path") or tag or ""
        if not _is_ir_dir(path):
            return {"backend": "openvino", "model": path, "accelerated": True,
                    "reason": (f"OpenVINO was chosen but {path or 'no directory'} "
                               f"is not a converted model - it needs a folder "
                               f"containing {IR_MARKER}.")}
        return {"backend": "openvino", "model": path, "accelerated": True,
                "reason": f"OpenVINO on {os.path.basename(path)}, as configured."}

    fallback = {"backend": "ollama", "model": tag, "accelerated": False}

    card = intel_gpu()
    if not card:
        return {**fallback, "reason": "No Intel GPU, so Ollama it is."}

    if not genai_available():
        return {**fallback,
                "reason": (f"{card} is here and would run narration several "
                           f"times faster, but openvino-genai is not installed. "
                           f"Install it to use the GPU for narration.")}

    configured = config.get("openvino_model_dir") or ""
    if _is_ir_dir(configured):
        note = ""
        if tag and not same_model(tag, configured):
            # Said out loud rather than done quietly: this is a different model
            # from the one the picker names, and its descriptions will differ.
            note = (f" This is not {tag}, which is what the model picker still "
                    f"shows - clear openvino_model_dir to go back to it.")
        return {"backend": "openvino", "model": configured, "accelerated": True,
                "reason": (f"OpenVINO on {card} with "
                           f"{os.path.basename(configured)}, from "
                           f"openvino_model_dir." + note)}

    models = converted_models()
    if not models:
        return {**fallback,
                "reason": (f"{card} is here and openvino-genai is installed, "
                           f"but no converted model was found. OpenVINO needs a "
                           f"model in its own format rather than the one Ollama "
                           f"keeps, which is a one-off conversion - see "
                           f"docs/INTEL-GPU.md. Narrating with Ollama instead.")}

    matches = [m for m in models if same_model(tag, m)]
    if matches:
        # An Instruct and a Thinking build of the same weights answer to the
        # same tag and are not interchangeable: one reasons before answering and
        # one does not, which is the whole reason a thinking model gets chosen.
        # `same_model` cannot tell them apart, because their names differ only
        # in a word it strips. So the tie is broken on behaviour instead, from
        # what each side already publishes - Ollama lists `thinking` among a
        # tag's capabilities, and a converted model's chat template carries the
        # tags. Name order picked silently between them once, and quietly turned
        # a reasoning narrator into a plain one.
        others = ""
        wanted = tag_reasons(tag) if len(matches) > 1 else None
        if wanted is not None:
            fitting = [m for m in matches if dir_reasons(m) == wanted]
            if fitting:
                skipped = [m for m in matches if m not in fitting]
                matches = fitting
                if skipped:
                    others = (f" Chosen because {tag} "
                              f"{'reasons' if wanted else 'does not reason'} and "
                              f"so does{'' if wanted else ' not'} this build, "
                              f"over "
                              + ", ".join(os.path.basename(m) for m in skipped)
                              + ".")
        match = matches[0]
        if not others and len(matches) > 1:
            rest = ", ".join(os.path.basename(m) for m in matches[1:])
            others = (f" {len(matches)} converted builds answer to {tag} and "
                      f"nothing here could tell them apart; this one was taken "
                      f"by name order, over {rest}. Set openvino_model_dir to "
                      f"choose.")
        return {"backend": "openvino", "model": match, "accelerated": True,
                "reason": (f"OpenVINO on {card} with {os.path.basename(match)} "
                           f"- the same model as {tag}, on the GPU, including "
                           f"the image encoding that Ollama leaves on the "
                           f"CPU." + others)}

    # Converted models exist but none is the one chosen. Substituting would
    # change the answers, not just the speed, so it is offered rather than done.
    names = ", ".join(os.path.basename(m) for m in models)
    return {**fallback,
            "reason": (f"{card} is here, but nothing converted matches {tag} "
                       f"(found: {names}). Convert {tag}, or point "
                       f"openvino_model_dir at one of those to use it instead. "
                       f"Narrating with Ollama for now.")}
