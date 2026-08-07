"""A paragraph per chapter: what it looks like was happening, in order.

Everything else in this report is arithmetic. A chapter can say it is 90%
speech, that a class is on screen for 72% of its detected seconds, and that two
words were said here and nowhere else -- and a person reading all three still
cannot tell you what happened in it. That gap is not a missing measurement. It
is the difference between description and narration, and no threshold over a
number closes it.

So this is the one part of the report written by a model, and it is built to be
obviously that. Three things keep it honest:

**It is given evidence, not asked to imagine.** The prompt for a chapter is the
sentences :mod:`modules.highlight_prose` already produced for it, the speech
transcribed inside it, and -- when the model can see -- a few frames from it.
Nothing is asked for that the material does not contain.

**It cannot introduce a figure.** Every number in a chapter block was computed
before this ran and is printed beside the paragraph. A reader who doubts a
sentence has the quotes and the measurements directly underneath it, which is
the whole reason the paragraph sits inside the chapter rather than replacing it.

**It is labelled, in the record and on the page.** ``story`` is stored next to
the model that wrote it, and the section says a model wrote it. A reading that
is presented as a measurement is worse than no reading; a reading that says what
it is costs the reader nothing to discount.

The chapters are told in order, each one handed a sentence of what preceded it,
because that is the difference between a list of scene descriptions and
something that reads continuously. It is also the one place a model can go
wrong cheaply and invisibly -- a wrong guess in chapter three becomes the
premise of chapter four -- so what is passed forward is deliberately short.
"""
from __future__ import annotations

import base64
import datetime as _dt
import os
from typing import Callable, Mapping, Optional, Sequence

# The model is being hired to do what the rest of the report refuses to do, so
# the licence has to be explicit -- and bounded in the same breath. Rule 2 is
# the load-bearing one: the page prints figures around this paragraph, and a
# paragraph that restates them slightly wrong is worse than one that has none.
STORY_SYSTEM_PROMPT = (
    "You are describing what happens in one stretch of a video, for someone who "
    "cannot watch it. You are given measurements another tool made, everything "
    "that was said in the stretch, and sometimes frames from it.\n"
    "Rules:\n"
    "1. Write plain continuous prose, past tense, 2-4 sentences. No headings, "
    "no bullet points, no preamble, no restating the question.\n"
    "2. Never state a number, a percentage or a timestamp. They are printed "
    "beside your paragraph already.\n"
    "3. Describe what is happening -- who is present, what they are doing, what "
    "is being said and how it changes. That is what is being asked for.\n"
    "4. Stay with what the material shows. Where you are inferring rather than "
    "reading something off, say so plainly ('appears to', 'seems to be'). Do "
    "not invent events, names or places that nothing mentions.\n"
    "5. If the material is too thin to say what happened, say that in one "
    "sentence instead of filling the space.\n"
    "6. Expression labels come from a five-class classifier that cannot tell a "
    "performed expression from a felt one. Do not treat them as feelings.\n"
    "7. Continue naturally from the previous stretch when one is given, without "
    "recapping it."
)

STORY_TASK = ("Say what happened in this stretch of the video, as if telling "
              "someone who has not seen it.")

# A paragraph, not an essay. Each chapter's block already carries its
# measurements and quotes; the prose is there to connect them, and at four
# sentences a model starts restating the list it was given.
STORY_TOKENS = 260

# Higher than the advisor's 0.3 for the same reason the reading is: this asks
# what something looked like, and at low temperature the answer is the safest
# possible sentence about any video, which is no answer.
STORY_TEMPERATURE = 0.8

# Per-chapter prompt budget. Much smaller than the whole-report prompt because
# there are as many of these as there are chapters, and a local model's minute
# per call is the real cost of this feature.
MAX_CHAPTER_CHARS = 9000

# How much of the previous chapter is carried forward. One paragraph's worth of
# context makes the run read continuously; more, and a model starts summarising
# the summary instead of describing what is in front of it -- and any error in
# it compounds down the chain.
CARRY_CHARS = 400

# Frames offered to a model that can see, spread across the chapter. Three
# covers a stretch's beginning, middle and end without the encode cost of a
# contact sheet, and vision models degrade quickly past a handful of images.
FRAMES_PER_CHAPTER = 3

# Longest edge of a frame handed to the model. Large enough to make out what is
# happening, small enough that three of them do not dominate the context.
FRAME_WIDTH = 512


def _chapter_facts(chapter: Mapping) -> list[str]:
    """The measured sentences for one chapter, as the page shows them."""
    try:
        from modules.highlight_prose import describe_chapter
        return list(describe_chapter(chapter))
    except Exception as exc:                       # pragma: no cover - defensive
        print(f"⚠️ Chapter facts skipped: {exc}")
        return []


def _checks_here(report: Mapping, chapter: Mapping) -> list:
    """Claims under test that were spoken inside this chapter, as told lines.

    Filed by *where the claim was said*, not by where its rule fired. The two
    are usually different stretches — somebody describes in the last five
    minutes something that happened in the middle — and the paragraph that
    should weigh the evidence is the one covering the sentence, because that is
    the paragraph a reader is reading when they wonder whether it is true.
    """
    start, end = float(chapter.get("start") or 0), float(chapter.get("end") or 0)
    lines = []
    for check in (report.get("checks") or []):
        at = check.get("claim_at")
        if at is None or not (start <= float(at) < end):
            continue
        claim = str(check.get("claim") or "").strip()
        state = str(check.get("state") or "")
        if state == "fired":
            outcome = (f"That arrangement was found in "
                       f"{int(check.get('seconds') or 0)} second(s) of the "
                       f"video, at {_timestamp(check.get('first') or 0)}"
                       + (f" and {_timestamp(check.get('last') or 0)}"
                          if check.get("last") != check.get("first") else "")
                       + ".")
        elif state == "never_fired":
            outcome = ("That arrangement was looked for in this run and never "
                       "found. It may still have happened out of the "
                       "detector's view.")
        else:
            outcome = ("The rule meant to look for it is not loaded, so this "
                       "run says nothing about it either way.")
        lines.append(f'- Someone said: "{claim}". A rule '
                     f'({check.get("label") or check.get("rule")}) was added to '
                     f'test it. {outcome} Say whether what you can see here '
                     f'agrees with what was said, and if the two do not match, '
                     f'say so plainly.')
    return lines


def _evidence_here(chapter: Mapping) -> list:
    """Things said in this stretch that the run has its own measurement of.

    The counterpart of :func:`_checks_here`, and the commoner case by far. That
    one answers a claim an *earlier* run went away and added a rule for; this
    one answers a claim the current run can already settle, because whatever was
    named happens to be something the detector was labelling all along.

    Told as evidence with an explicit ask rather than left in the measured
    facts above, because a model handed a figure treats it as background and a
    model handed a question answers it. The question is the feature: a narration
    that repeats what a speaker said about a moment has added nothing, and the
    same narration saying the run found that moment somewhere else, or barely at
    all, is the thing a transcript on its own cannot produce.

    The caution about what a match means is stated once, in the section header,
    rather than per row — a model reading it five times starts hedging every
    sentence, and the point is to get a verdict, not a disclaimer.
    """
    from modules.spoken_evidence import MAX_PER_CHAPTER, MIN_LEVEL_DELTA_DB

    lines = []
    for row in (chapter.get("spoken_evidence") or [])[:MAX_PER_CHAPTER]:
        name = str(row.get("name") or "")
        seconds = int(row.get("seconds") or 0)
        said = (f'Someone said at {row.get("timestamp", "")}: '
                f'"{str(row.get("quote") or "").strip()}". '
                f'This run was labelling {name} throughout the video.')
        if not seconds:
            lines.append(f"- {said} It was never labelled anywhere in the "
                         f"file, so what was said here has nothing in this run "
                         f"agreeing with it. Say so.")
            continue

        facts = [f"{name} was labelled in {seconds} second(s) of the video"]
        share = row.get("video_share_pct")
        if share:
            facts[-1] += f", {float(share):.0f}% of everything it detected"
        densest = row.get("densest_chapter") or {}
        if densest:
            facts.append(f"most of it in chapter {densest.get('number')} at "
                         f"{densest.get('timestamp')}, where it holds "
                         f"{float(densest.get('share_pct') or 0):.0f}% of the "
                         f"stretch")
        elif row.get("first"):
            facts.append(f"first labelled at {row['first']}, last at "
                         f"{row.get('last')}")
        here = row.get("here_share_pct")
        if here is not None:
            facts.append(f"{float(here):.0f}% of this stretch")
        loudest = (row.get("level") or {}).get("loudest") or {}
        if loudest:
            delta = loudest.get("vs_video_db")
            fact = f"its loudest second is {loudest.get('timestamp')}"
            if delta is not None and abs(float(delta)) >= MIN_LEVEL_DELTA_DB:
                fact += (f", {abs(float(delta)):.0f} dB "
                         f"{'above' if float(delta) >= 0 else 'below'} the "
                         f"video's median level")
            alongside = loudest.get("with") or []
            if alongside:
                fact += (f", with {', '.join(str(a) for a in alongside)} also "
                         f"labelled at that second")
            facts.append(fact)
        clips = row.get("clips") or []
        if clips:
            facts.append(f"{len(clips)} of the kept clips overlap it"
                         if len(clips) > 1 else "one kept clip overlaps it")
        # Not capitalised: the first word of the list is a class name, and
        # those are the user's own, where case can be part of the name.
        lines.append(f"- {said} " + "; ".join(facts) +
                     ". Say whether that bears out what was said, and where it "
                     "does not, say so plainly.")
    return lines


def _timestamp(seconds) -> str:
    try:
        seconds = max(0, int(seconds))
    except (TypeError, ValueError):
        return "?"
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def _dialogue_block(chapter: Mapping, budget: int) -> str:
    """Everything said in the chapter, trimmed from the middle if it must be.

    From the middle rather than the end, because a stretch's opening and closing
    lines are what place it -- how something started and where it had got to.
    Cutting the tail instead would describe the first third of every long
    chapter and call it the chapter.
    """
    lines = list(chapter.get("dialogue") or chapter.get("quotes") or [])
    if not lines:
        return ""
    rendered = []
    for line in lines:
        who = f"{line['speaker']}: " if line.get("speaker") else ""
        rendered.append(f"{who}{line.get('text', '')}")
    text = "\n".join(rendered)
    if len(text) <= budget:
        return text
    half = max(1, budget // 2)
    head, tail = text[:half], text[-half:]
    return (f"{head}\n[... middle of the stretch omitted for length ...]\n"
            f"{tail}")


def chapter_prompt(report: Mapping, chapter: Mapping,
                   previous: Optional[str] = None,
                   max_chars: int = MAX_CHAPTER_CHARS) -> str:
    """Everything the model is told about one chapter."""
    video = report.get("video") or {}
    parts = [
        "## The stretch",
        f"Chapter {chapter.get('number', '?')} of "
        f"{len(report.get('chapters') or [])}, running "
        f"{float(chapter.get('duration') or 0) / 60:.0f} minutes, from "
        f"{chapter.get('timestamp', '')} of a "
        f"{float(video.get('duration') or 0) / 60:.0f}-minute video.",
        "",
    ]

    facts = _chapter_facts(chapter)
    if facts:
        parts += ["## What was measured here", *(f"- {f}" for f in facts), ""]

    # A claim an earlier run added a rule to test, when that claim was made in
    # this stretch. This is the point of the whole loop: the run that finally
    # has the signal is the one that can say whether it agrees, and it can only
    # do that if it is told which sentence it is answering.
    outstanding = _checks_here(report, chapter)
    if outstanding:
        parts += ["## A claim made here that a later rule was added to test",
                  *outstanding, ""]

    # Above the transcript, because it is about a specific line in it and a
    # model reading the transcript afterwards will find that line and weigh it.
    # Below the measurements, because those describe *this* stretch and these
    # figures describe the whole video.
    evidence = _evidence_here(chapter)
    if evidence:
        parts += ["## Something said here that this run measured for itself",
                  "The match is between the words spoken and the name of "
                  "something the detector was labelling — nothing more. If the "
                  "line reads as being about something else, say that instead "
                  "of forcing the two together.",
                  *evidence, ""]

    if previous:
        parts += ["## What the previous stretch was",
                  previous.strip()[:CARRY_CHARS], ""]

    # Last, and given whatever budget is left, because it is the section that
    # actually says what is going on. The measurements above are a page of
    # numbers about the same stretch; the words are the stretch.
    head = "\n".join(parts)
    spare = max(500, max_chars - len(head) - len(STORY_TASK) - 200)
    said = _dialogue_block(chapter, spare)
    if said:
        parts += ["## What was said here, in order", said, ""]
    else:
        parts += ["## What was said here", "Nothing was transcribed as speech "
                  "in this stretch.", ""]

    parts += ["## Task", STORY_TASK]
    return "\n".join(parts)


def frames_from_video(video_path: str, count: int = FRAMES_PER_CHAPTER):
    """A ``(start, end) -> [base64 jpeg]`` sampler, or ``None`` without a video.

    Returned as a closure so the caller opens the file once per chapter and the
    rest of this module never learns what OpenCV is. ``None`` when the source is
    not beside the report, which is the ordinary case for a report that was
    moved -- and it costs only the pictures, not the narration.
    """
    if not video_path or not os.path.exists(video_path):
        return None
    try:
        import cv2
    except Exception as exc:                       # pragma: no cover - defensive
        print(f"⚠️ Chapter frames skipped: {exc}")
        return None

    def sample(start: float, end: float) -> list:
        cap = cv2.VideoCapture(video_path)
        out = []
        try:
            if not cap.isOpened():
                return out
            span = max(0.0, float(end) - float(start))
            # Inset from both edges: a chapter boundary falls on a cut, and the
            # frame either side of a cut is the least representative one in the
            # stretch.
            for i in range(count):
                at = float(start) + span * (i + 0.5) / count
                cap.set(cv2.CAP_PROP_POS_MSEC, at * 1000.0)
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                h, w = frame.shape[:2]
                if w > FRAME_WIDTH:
                    frame = cv2.resize(
                        frame, (FRAME_WIDTH, max(1, int(h * FRAME_WIDTH / w))))
                ok, buf = cv2.imencode(".jpg", frame,
                                       [int(cv2.IMWRITE_JPEG_QUALITY), 72])
                if ok:
                    out.append(base64.b64encode(buf.tobytes()).decode("ascii"))
        finally:
            cap.release()
        return out

    return sample


def _tell_one(llm, prompt: str, images: Optional[Sequence[str]]) -> str:
    """One call, with pictures when the model takes them.

    Falls back to text on any image-related failure rather than losing the
    paragraph: a model that cannot see still has the transcript, which is most
    of what the answer rests on anyway.
    """
    from modules.advisor import _generate

    if images:
        try:
            return llm.generate(prompt, system=STORY_SYSTEM_PROMPT,
                                max_tokens=STORY_TOKENS,
                                temperature=STORY_TEMPERATURE,
                                images=list(images))
        except Exception as exc:
            print(f"⚠️ Chapter frames not used ({exc}); telling from text alone")
    return _generate(llm, prompt, STORY_SYSTEM_PROMPT, STORY_TOKENS,
                     STORY_TEMPERATURE)


def tell(report: Mapping,
         *,
         llm,
         frames_fn: Optional[Callable] = None,
         log_fn=print,
         cancel_fn: Optional[Callable[[], bool]] = None) -> list[dict]:
    """Narrate every chapter in order. Returns the chapters, story attached.

    One call per chapter, which on a local model is the slowest thing this
    report does — hence ``log_fn`` per chapter and ``cancel_fn`` between them.
    A chapter whose call fails keeps its measurements and loses only its
    paragraph; the run continues, because sixteen chapters minus one is still
    the thing the user asked for.
    """
    chapters = [dict(ch) for ch in (report.get("chapters") or [])]
    if not chapters or llm is None:
        return chapters

    previous = None
    for index, ch in enumerate(chapters, start=1):
        if cancel_fn is not None and cancel_fn():
            log_fn(f"⏹️ Stopped after {index - 1} of {len(chapters)} chapters.")
            break
        log_fn(f"📖 Telling chapter {index} of {len(chapters)}"
               f" ({ch.get('timestamp', '')})…")
        images = None
        if frames_fn is not None:
            try:
                images = frames_fn(float(ch["start"]), float(ch["end"]))
            except Exception as exc:
                print(f"⚠️ Frames for chapter {index} failed: {exc}")
        try:
            text = (_tell_one(llm, chapter_prompt(report, ch, previous),
                              images) or "").strip()
        except Exception as exc:
            log_fn(f"⚠️ Chapter {index} could not be told: {exc}")
            continue
        if not text:
            continue
        ch["story"] = text
        # Only a told chapter carries forward. A skipped one would otherwise
        # hand the next chapter the one before it as if they were adjacent.
        previous = text
    return chapters


def tell_report_file(json_path: str,
                     *,
                     llm,
                     model_name: Optional[str] = None,
                     use_frames: bool = True,
                     log_fn=print,
                     cancel_fn: Optional[Callable[[], bool]] = None) -> int:
    """Narrate the chapters of a report on disk. Returns how many were told.

    Re-renders the page from the updated record rather than patching it, for the
    same reason :func:`modules.advisor.summarise_report_file` does: the record
    and the page must not be able to disagree.
    """
    import json

    from modules.highlight_report import render_html

    with open(json_path, encoding="utf-8") as fh:
        report = json.load(fh)

    frames_fn = None
    if use_frames:
        frames_fn = frames_from_video(str((report.get("video") or {}).get("path")
                                          or ""))
        if frames_fn is None:
            log_fn("ℹ️ Source video not found beside the report — telling from "
                   "the transcript and the measurements alone.")

    chapters = tell(report, llm=llm, frames_fn=frames_fn, log_fn=log_fn,
                    cancel_fn=cancel_fn)
    told = [ch for ch in chapters if ch.get("story")]
    if not told:
        return 0

    report["chapters"] = chapters
    # Stamped once for the run rather than per chapter: they were all told by
    # the same model in the same pass, and sixteen copies of the same string is
    # noise in a record people read.
    report["chapter_story"] = {
        "model": str(model_name or ""),
        "at": _dt.datetime.now().isoformat(timespec="seconds"),
        "chapters": len(told),
        "with_frames": bool(frames_fn),
    }
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)

    html_path = json_path[:-5] + ".html" if json_path.endswith(".json") else None
    if html_path and os.path.exists(html_path):
        with open(html_path, "w", encoding="utf-8") as fh:
            fh.write(render_html(report))
    return len(told)
