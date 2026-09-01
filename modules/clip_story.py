"""A paragraph per kept clip, read off the clip's own frames.

:mod:`modules.chapter_story` closes this gap for a *chapter*: a stretch of
several minutes, told mostly from what was said in it. That works, and on a
video with a transcript it works well enough that the clip cards need nothing —
each one already carries the lines spoken over it, and "said here" turns out to
be most of what a reader wanted when they asked what a moment was.

On footage with no speech there is nothing in that slot at all, and the
difference between the two reports is stark. Everything under "The moments, in
order" is still true — the movement burst, the level against the video's median,
the classifier's reading — and none of it says what is on screen. A page that
can describe a moment in six measured sentences without once mentioning what is
in the picture is not a page that has explained the moment.

So this asks the model that *can* see. One call per kept clip, a few frames
spread across it, and the sentences the report already wrote about it — the same
three commitments :mod:`modules.chapter_story` makes, for the same reasons:

**It is given evidence, not asked to imagine.** The frames, the measured
sentences, the detector's labels, anything said, and any notes a narration pass
already wrote over this span. Nothing is asked for that the material does not
contain.

**It cannot introduce a figure.** Every number on the card was computed before
this ran and is printed under the paragraph. The reading sits above the evidence
it was written from, inside the same card, so a reader who doubts a sentence
does not have to leave it.

**It is labelled, in the record and on the page.** ``story`` on the segment,
``clip_story`` on the report, and the words "read from this clip" on the card.

Where it differs from the chapter pass
--------------------------------------

A chapter is minutes long and is told in order, each one handed what came
before, because a run of chapters should read continuously. A clip is thirty
seconds and its neighbour is often twenty minutes away, so there is nothing to
continue *from* — the carry exists here only to stop fourteen cards opening with
the same sentence, and it is the one thing the previous clip is used for.

The other difference is what the frames are for. A chapter's frames are a
sanity check on a paragraph the transcript could mostly have written; here they
are the whole content, and the prompt says so. Frames are sampled across the
clip rather than at its peak second precisely so the answer can be about what
*changes* — which is the one thing the thumbnail already on the card cannot show,
and generally what the clip was picked for.

Cost
----

One vision call per kept clip: fourteen clips is a minute or two on a local
model, against the hours a whole-video narration costs. That ratio is the reason
this exists as its own pass rather than as a filter over
:mod:`modules.moment_narration` — but when a narration *has* been run, its notes
for these seconds are handed over as material, because a note written from a
frame beats a second guess at the same frame.
"""
from __future__ import annotations

import datetime as _dt
import os
from typing import Callable, Mapping, Optional, Sequence

# The licence and its bounds, as in `chapter_story` — with rule 3 doing the work
# that rule is not needed for there. The frames are the content of this answer,
# and a model handed four pictures and a page of measurements will otherwise
# write about the measurements, which the card already says better.
CLIP_SYSTEM_PROMPT = (
    "You are describing one short clip from a video, for someone who cannot "
    "watch it. You are given a few frames taken across the clip, measurements "
    "another tool made about it, and whatever was said in it.\n"
    "Rules:\n"
    "1. Write plain continuous prose, present tense, 2-4 sentences. No "
    "headings, no bullet points, no preamble, no restating the question.\n"
    "2. Never state a number, a percentage or a timestamp. They are printed "
    "beside your paragraph already.\n"
    "3. Say what is in the frames: who is present, where they are, what they "
    "are doing. This is what the rest of the page cannot say, so it is what is "
    "being asked for.\n"
    "4. Where you are given more than one frame they are in order, a few "
    "seconds apart. Say what changes between them — that is what the clip was "
    "picked for, and what a single still cannot show.\n"
    "5. Stay with what the material shows. Where you are inferring rather than "
    "reading something off, say so plainly ('appears to', 'seems to be'). Do "
    "not invent events, names or places that nothing mentions.\n"
    "6. If the frames are too dark, too close or too few to make out, say that "
    "in one sentence instead of filling the space.\n"
    "7. Expression labels come from a five-class classifier that cannot tell a "
    "performed expression from a felt one. Do not treat them as feelings.\n"
    "8. When a previous clip is given, do not recap it and do not open the same "
    "way; say what is different here.\n"
    "9. When the run has flagged something here -- a detected label or a "
    "composition rule -- neither ignore it nor repeat it as fact. Say whether "
    "what you can see supports it, and attribute it to the run: 'the run "
    "flagged X here, and the frames show / do not show Y'. If the frames "
    "neither support nor contradict it, say exactly that. You were told it "
    "fired, which is not the same as having seen it, so it is never your own "
    "observation. This one sentence is worth more than another describing the "
    "frames, because it is the only part of the page that can disagree with "
    "the detector."
)

# The captioner's copy of the same brief — see
# :data:`modules.chapter_story.CAPTIONER_SYSTEM_PROMPT` for why a second one
# exists at all. Rule 9 survives the flattening as its own paragraph rather than
# a clause: it is the only sentence on the page allowed to disagree with the
# detector, and it is the first thing a shorter prompt loses.
CAPTIONER_SYSTEM_PROMPT = (
    "Describe one short clip from a video, for someone who cannot watch it. "
    "You are given a few frames taken across it, measurements another tool "
    "made, and whatever was said.\n"
    "Say what is in the frames: who is present, where they are, what they are "
    "doing. That is what the rest of the page cannot say.\n"
    "The frames are in order, a few seconds apart. Say what changes between "
    "them.\n"
    "Write two to four plain sentences in the present tense. No headings, no "
    "lists, no preamble.\n"
    "Never write a number, a percentage or a time. They are printed beside "
    "your words already.\n"
    "Where you are guessing rather than reading something off, say so — "
    "'appears to', 'seems to be'. Invent nothing that the material does not "
    "mention.\n"
    "Treat any expression label you are given as a guess by another tool, not "
    "as what someone felt.\n"
    "When you are shown the clip before this one, do not recap it and do not "
    "open the same way. Say what is different here.\n"
    "When you are told the run flagged something here, you were told it fired "
    "— you did not see it fire. Say whether the frames support it and say who "
    "claimed it: 'the run flagged X, and the frames show / do not show Y'. If "
    "the frames neither support nor contradict it, say exactly that. Never "
    "repeat it as your own observation.\n"
    "If the frames are too dark, too close or too few to make out, say only: "
    "Too little to see here."
)

CLIP_TASK = ("Say what this clip shows, as if telling someone who has not seen "
             "it.")

# Shorter than a chapter's. A clip is thirty seconds with one thing happening in
# it, and past three sentences a model starts narrating the measurements it was
# handed rather than the picture.
CLIP_TOKENS = 200

CLIP_TEMPERATURE = 0.8

# Per-clip prompt budget. Smaller than a chapter's for the same reason a clip is
# smaller than a chapter: there is simply less to say, and the frames are most
# of the payload.
MAX_CLIP_CHARS = 5000

# How much of the previous clip is carried. Half a chapter's, because this is
# only here to break the opening-sentence lock — a clip has no continuity with
# its neighbour worth preserving, and more context is more chance of an error
# from one card being restated as fact on the next.
CARRY_CHARS = 200

# Spread across the clip, not clustered at the peak. Four rather than the
# chapter pass's three: a thirty-second span sampled four times is roughly every
# seven seconds, which is close enough that a change between two of them is
# legible as a change rather than as two unrelated pictures.
FRAMES_PER_CLIP = 4

# Notes from an earlier narration that fall inside a clip. Enough to establish
# what was already observed across the span, few enough that they do not become
# the answer — this pass is looking at the frames, not summarising a log.
MAX_NOTES = 6

# Lines of speech offered per clip. The card prints three and says how many more
# there were; the prompt can afford all of a thirty-second clip's dialogue.
MAX_SPEECH_LINES = 20


def _clip_facts(report: Mapping, entry: Mapping) -> list:
    """The measured sentences for one clip, as the card shows them.

    Read through :mod:`modules.highlight_report` rather than assembled here, so
    the prompt and the page cannot end up quoting different figures for one run
    — the same rule :func:`modules.chapter_story._chapter_facts` follows.
    """
    try:
        from modules.highlight_prose import clip_sections
        from modules.highlight_report import _segment_readings

        arc = report.get("expression_arc") or {}
        valence = float((arc.get("valence") or {}).get("mean_all_read") or 0.0)
        peers = [float(e.get("score") or 0.0)
                 for e in (report.get("segments") or [])]
        reading = _segment_readings(report).get(entry.get("index"))
        lines = []
        for heading, sentences in clip_sections(entry, reading, valence, peers):
            for sentence in sentences:
                lines.append(f"{heading}: {sentence}")
        return lines
    except Exception as exc:                       # pragma: no cover - defensive
        print(f"⚠️ Clip facts skipped: {exc}")
        return []


def _labels_here(entry: Mapping) -> list:
    """What the detector, the composition rules and the action model called it.

    Named as three separate sources rather than merged into one list of words,
    because they carry very different weight and a model given them flat treats
    the 0.01-confidence action exactly as it treats the object that was measured
    for the whole video.
    """
    lines = []
    objects = [str(o) for o in (entry.get("objects") or [])]
    if objects:
        lines.append("The object detector labelled these at the peak second: "
                     + ", ".join(objects) + ".")
    events = [str(v) for v in (entry.get("events") or [])]
    if events:
        lines.append("The composition rules recognised: " + ", ".join(events)
                     + ". These are combinations of the labels above, not a "
                     "separate observation.")
    actions = entry.get("actions") or []
    if actions:
        named = ", ".join(f'{a.get("name")} (confidence '
                          f'{float(a.get("confidence") or 0.0):.2f})'
                          for a in actions)
        lines.append(f"The action model's best guesses: {named}. Its labels "
                     f"come from a fixed everyday-activity vocabulary, so a low "
                     f"confidence usually means it has no word for what is "
                     f"happening. Ignore one that does not match the frames.")
    return lines


def _notes_here(report: Mapping, entry: Mapping) -> list:
    """Notes an earlier narration pass wrote inside this clip, if there is one.

    Free when it exists and absent when it does not, which is the ordinary case.
    Offered as observation rather than as conclusion: each was written from one
    frame by a model that could not see the others, and the whole point of
    handing them over together is that this call can see the span they cover.
    """
    try:
        # Looked up by name rather than imported outright, because this module
        # is shared verbatim with an edition that has no per-frame narration and
        # therefore no such file. A static `from modules import narration_notes`
        # would be a promise this file cannot keep there — and the import
        # checker is right to fail it, so the fix is to stop making the promise
        # rather than to teach the checker about guards.
        #
        # Its absence is the edition boundary, not a fault, so it says nothing.
        # Caught separately from the rest: a *real* fault below still gets its
        # warning, which is the whole value of not widening this except.
        import importlib

        try:
            narration_notes = importlib.import_module("modules.narration_notes")
        except ImportError:
            return []

        path = str((report.get("video") or {}).get("path") or "")
        entries = narration_notes.load(path)
        if not entries:
            return []
        inside = narration_notes.within(
            entries, [(float(entry.get("start") or 0.0),
                       float(entry.get("end") or 0.0))])
        return [" ".join(str(e.get("text") or "").split())
                for e in inside[:MAX_NOTES]]
    except Exception as exc:                       # pragma: no cover - defensive
        print(f"⚠️ Narration notes skipped: {exc}")
        return []


def _speech_here(entry: Mapping) -> list:
    """What was said over the clip, speaker first where one was identified."""
    said = entry.get("speech") or {}
    lines = []
    for line in (said.get("lines") or [])[:MAX_SPEECH_LINES]:
        who = f'{line["speaker"]}: ' if line.get("speaker") else ""
        text = str(line.get("text") or "").strip()
        if text:
            lines.append(f"{who}{text}")
    return lines


def clip_prompt(report: Mapping, entry: Mapping,
                previous: Optional[str] = None,
                max_chars: int = MAX_CLIP_CHARS,
                frame_count: int = 0) -> str:
    """Everything the model is told about one clip.

    ``frame_count`` is how many frames the model will actually be shown, which
    is not always how many were sampled — see :func:`_frames_delivered`. It is
    stated in the prompt because rule 4 asks what changes between them, and a
    model handed one picture and told it has four will answer the question it
    was asked rather than the one it can see.
    """
    segments = report.get("segments") or []
    video = report.get("video") or {}
    if frame_count > 1:
        seen = f"You have {frame_count} frames from across this clip, in order."
    elif frame_count == 1:
        seen = "You have one frame, from the middle of this clip."
    else:
        seen = "You have no frames for this clip."
    parts = [
        "## The clip",
        f"Clip {entry.get('index', '?')} of {len(segments)}, "
        f"{float(entry.get('duration') or 0):.0f} seconds, at "
        f"{entry.get('range') or entry.get('timestamp') or ''} of a "
        f"{float(video.get('duration') or 0) / 60:.0f}-minute video.",
        seen,
        "",
    ]

    facts = _clip_facts(report, entry)
    if facts:
        parts += ["## What was measured here",
                  "Context for what you are looking at. Do not repeat these — "
                  "they are printed under your paragraph already.",
                  *(f"- {f}" for f in facts), ""]

    labels = _labels_here(entry)
    if labels:
        # Asked for explicitly, because the measurements section above says not
        # to repeat what the page already prints and a model applies that to
        # everything it is handed. A run flagged something here and the
        # paragraph said nothing about it, which is the one omission that makes
        # this section worth reading at all.
        parts += ["## What was labelled here",
                  "Weigh these against the frames and say so — rule 9. Whether "
                  "you can see what was flagged is the question; agreeing with "
                  "it is not the answer.",
                  *labels, ""]

    notes = _notes_here(report, entry)
    if notes:
        parts += ["## Notes written over this span earlier",
                  "Each was written from a single frame of this span by "
                  "someone who could not see the others, and not necessarily "
                  "from the frames you have. You can see the span. Use them, "
                  "and where they disagree with what is in front of you, go "
                  "with the frames.",
                  *(f"- {n}" for n in notes), ""]

    if previous:
        parts += ["## What the previous clip was",
                  previous.strip()[:CARRY_CHARS], ""]

    # Last, and given whatever budget is left. On the footage this pass exists
    # for the section is empty and says so — which is itself worth telling the
    # model, so it does not go looking for dialogue to account for.
    head = "\n".join(parts)
    spare = max(400, max_chars - len(head) - len(CLIP_TASK) - 200)
    speech = "\n".join(_speech_here(entry))[:spare]
    if speech:
        parts += ["## What was said here, in order", speech, ""]
    else:
        parts += ["## What was said here",
                  "Nothing was transcribed as speech in this clip, so the "
                  "frames are all there is to go on.", ""]

    parts += ["## Task", CLIP_TASK]
    return "\n".join(parts)


def _takes_a_list(query) -> bool:
    """Whether this ``query`` accepts several frames or only ``frame_base64``.

    Asked rather than assumed because the answer changed: ``LLMModule.query``
    took one picture for most of its life, and an object of that vintage handed
    ``frames=`` raises ``TypeError`` — which would cost the clip its paragraph
    for a reason that has nothing to do with the footage.
    """
    import inspect

    try:
        return "frames" in inspect.signature(query).parameters
    except (TypeError, ValueError):                # pragma: no cover - builtins
        return False


def _frames_delivered(llm, images: Optional[Sequence[str]]) -> int:
    """How many of these frames this object will actually put in front of a model.

    Not the same as how many were sampled, and the gap is a trap worth naming.
    Sampling four and sending one is fine; *saying* four while sending one is
    not, because the answer then describes motion nobody showed the model —
    which is why this and :func:`_read_one` have to agree, and why the prompt
    is told the number this returns rather than the number that was sampled.
    """
    frames = [i for i in (images or []) if i]
    if not frames:
        return 0
    if hasattr(llm, "generate"):
        return len(frames)
    if hasattr(llm, "query"):
        return len(frames) if _takes_a_list(llm.query) else 1
    return 0


def system_prompt_for(model_name: Optional[str]) -> str:
    """The brief this model can actually receive."""
    from modules.llm_models import is_captioner

    return CAPTIONER_SYSTEM_PROMPT if is_captioner(model_name) \
        else CLIP_SYSTEM_PROMPT


def _read_one(llm, prompt: str, images: Sequence[str],
              system: Optional[str] = None) -> str:
    """One vision call, on whichever interface the object offers.

    Both are handled explicitly, and anything else is an error — the same
    reasoning as :func:`modules.moment_narration._caption_one`, and for the
    same reason it had to be written there. Reaching for ``generate`` inside a
    ``try`` and falling back to text on ``AttributeError`` looks like tolerance
    of a missing feature; against an ``LLMModule`` it *always* takes the
    fallback, and every paragraph in the run is then written from the
    measurements with one warning line at the top of the log.

    That failure is invisible in the output. A model asked to describe a clip
    it was never shown writes a fluent, plausible paragraph out of the figures
    it was handed, and nothing on the page distinguishes it from one written
    from the pictures. So there is no text-only path here at all: a clip that
    cannot be seen loses its paragraph and keeps its measurements.
    """
    frames = [i for i in (images or []) if i]
    if not frames:
        raise ValueError("no frames to read")
    system = system or CLIP_SYSTEM_PROMPT
    if hasattr(llm, "generate"):
        return llm.generate(prompt, system=system,
                            max_tokens=CLIP_TOKENS,
                            temperature=CLIP_TEMPERATURE,
                            images=list(frames))
    if hasattr(llm, "query"):
        if _takes_a_list(llm.query):
            return llm.query(prompt, system_prompt=system,
                             frames=list(frames), free_chat_mode=True,
                             max_tokens=CLIP_TOKENS,
                             temperature=CLIP_TEMPERATURE)
        # One frame only, and the middle one: the edges of a sampled span sit
        # closest to whatever bounded it, which for a clip is its own boundary.
        return llm.query(prompt, system_prompt=system,
                         frame_base64=frames[len(frames) // 2],
                         free_chat_mode=True, max_tokens=CLIP_TOKENS,
                         temperature=CLIP_TEMPERATURE)
    raise TypeError(
        f"{type(llm).__name__} offers neither generate() nor query()")


def tell(report: Mapping,
         *,
         llm,
         frames_fn: Optional[Callable] = None,
         model_name: Optional[str] = None,
         log_fn=print,
         cancel_fn: Optional[Callable[[], bool]] = None) -> list:
    """Read every kept clip. Returns the segments, ``story`` attached.

    A clip whose call fails keeps its measurements and loses only its
    paragraph, and the run continues — thirteen of fourteen is still the thing
    the user asked for. A clip whose *frames* fail is skipped for the same
    reason :func:`_read_one` has no text-only path: there is nothing to read.

    ``model_name`` picks the brief and nothing else — see
    :func:`modules.chapter_story.tell`.
    """
    from modules.chapter_story import trim_repetition

    segments = [dict(e) for e in (report.get("segments") or [])]
    if not segments or llm is None:
        return segments

    system = system_prompt_for(model_name)

    previous = None
    for position, entry in enumerate(segments, start=1):
        if cancel_fn is not None and cancel_fn():
            log_fn(f"⏹️ Stopped after {position - 1} of {len(segments)} clips.")
            break
        log_fn(f"🖼️ Reading clip {position} of {len(segments)}"
               f" ({entry.get('range', '')})…")
        images = None
        if frames_fn is not None:
            try:
                images = frames_fn(float(entry.get("start") or 0.0),
                                   float(entry.get("end") or 0.0))
            except Exception as exc:
                print(f"⚠️ Frames for clip {position} failed: {exc}")
        seen = _frames_delivered(llm, images)
        if not seen:
            log_fn(f"⚠️ Clip {position} has no frames to read; skipped.")
            continue
        try:
            text = (_read_one(llm,
                              clip_prompt(report, entry, previous,
                                          frame_count=seen),
                              images, system) or "").strip()
        except Exception as exc:
            log_fn(f"⚠️ Clip {position} could not be read: {exc}")
            continue
        if not text:
            continue
        trimmed = trim_repetition(text)
        if len(trimmed) < len(text):
            log_fn(f"✂️ Clip {position} repeated itself; kept what came before.")
            text = trimmed
        entry["story"] = text
        previous = text
    return segments


def tell_report_file(json_path: str,
                     *,
                     llm,
                     model_name: Optional[str] = None,
                     frames_fn: Optional[Callable] = None,
                     log_fn=print,
                     cancel_fn: Optional[Callable[[], bool]] = None) -> int:
    """Read the clips of a report on disk. Returns how many were read.

    Re-renders the page from the updated record rather than patching it, for the
    same reason :func:`modules.chapter_story.tell_report_file` does: the record
    and the page must not be able to disagree.

    ``frames_fn`` is derived from the record's own video path when not supplied.
    There is no way to ask for this pass *without* frames, because there is no
    such pass — see :func:`_read_one`.
    """
    import json

    from modules.chapter_story import frames_from_video
    from modules.highlight_report import media_src_for, render_html

    with open(json_path, encoding="utf-8") as fh:
        report = json.load(fh)

    if frames_fn is None:
        source = str((report.get("video") or {}).get("path") or "")
        frames_fn = frames_from_video(source, count=FRAMES_PER_CLIP)
    if frames_fn is None:
        # Refused rather than run: the whole content of these paragraphs is the
        # frames, so a run without them would spend the model time to produce
        # nothing the card does not already say — and would look, on the page,
        # exactly like a run that had worked.
        log_fn("⚠️ The source video is not where the report says it is, so "
               "there are no frames to read. Nothing was written — this pass "
               "has nothing to say without the pictures.")
        return 0

    segments = tell(report, llm=llm, frames_fn=frames_fn,
                    model_name=model_name, log_fn=log_fn, cancel_fn=cancel_fn)
    read = [e for e in segments if e.get("story")]
    if not read:
        return 0

    # Re-read before writing, and merge rather than replace. This pass takes
    # minutes, and the record on disk is live the whole time: the advisor's
    # summary and the chat's answers are written into the same file from the
    # app, by a user who is looking at the report *while* this runs. Writing
    # back the copy loaded at the start silently reverts whatever they did in
    # the meantime — which is not a hypothetical, it happened on the run this
    # was written for, and it looks from the outside exactly like the pass
    # having done nothing.
    #
    # Only what this pass owns is merged in: the per-clip paragraph, matched by
    # clip index, and the stamp. Anything else in the newer record wins.
    latest = report
    try:
        with open(json_path, encoding="utf-8") as fh:
            latest = json.load(fh)
    except Exception as exc:                       # pragma: no cover - defensive
        print(f"⚠️ Could not re-read {json_path} ({exc}); writing what was read")

    stories = {e.get("index"): e["story"] for e in read}
    written = 0
    for entry in (latest.get("segments") or []):
        story = stories.get(entry.get("index"))
        if story:
            entry["story"] = story
            written += 1
    if not written:
        # The clips on disk are not the clips that were read — a re-analysis
        # landed while this ran. Its report is the current one and this one
        # describes footage it no longer contains.
        log_fn("⚠️ The report was re-analysed while the clips were being read, "
               "so these readings describe a cut that no longer exists. "
               "Nothing was written.")
        return 0

    latest["clip_story"] = {
        "model": str(model_name or ""),
        "at": _dt.datetime.now().isoformat(timespec="seconds"),
        "clips": written,
    }
    report = latest
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)

    html_path = json_path[:-5] + ".html" if json_path.endswith(".json") else None
    if html_path and os.path.exists(html_path):
        # Through `media_src_for`, so re-rendering does not quietly strip the
        # per-clip players a full write put there.
        with open(html_path, "w", encoding="utf-8") as fh:
            fh.write(render_html(report, media_src=media_src_for(report,
                                                                 html_path)))
    return len(read)


def _main(argv=None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        prog="python -m modules.clip_story",
        description="Describe each kept clip of a report from its own frames.")
    parser.add_argument("report", help="the *_why.json written beside a cut")
    parser.add_argument("--backend", default="ollama",
                        choices=("ollama", "llama-cpp"))
    # Was llava-llama3, which is a generation behind and fails this job rather
    # than merely doing it worse. Measured over one 14-clip report against the
    # same frames and prompt: it opened paragraphs "This image captures...",
    # having processed one of the four frames it was given, which makes rule 4
    # unanswerable by construction; it described a moment with several people
    # as involving two, spending its length on flooring and furniture; and it
    # took 296 s against 212 s. See docs/advisor/models.md.
    parser.add_argument("--model", default="qwen2.5vl:7b",
                        help="a model that can see, current-generation; "
                             "a .gguf path for llama-cpp")
    parser.add_argument("--mmproj", default=None,
                        help="vision projector, for llama-cpp models that need "
                             "one beside the weights")
    args = parser.parse_args(argv)

    from modules.advisor import load_llm

    llm = load_llm(args.backend, args.model, mmproj=args.mmproj, vision=True)
    if llm is None:
        return 1
    read = tell_report_file(args.report, llm=llm,
                            model_name=f"{args.backend}/{args.model}")
    if not read:
        print("Nothing was read; the report is unchanged.")
        return 1
    with open(args.report, encoding="utf-8") as fh:
        total = len(json.load(fh).get("segments") or [])
    print(f"Read {read} of {total} clips.")
    return 0


if __name__ == "__main__":                         # pragma: no cover
    raise SystemExit(_main())
