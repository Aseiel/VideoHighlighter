"""The advisor: findings, the pages that explain them, and an optional narrator.

The split matters more than either half. ``highlight_advice`` computes *what* is
wrong from the report, deterministically. ``docs/advisor`` says what that means
and what to do about it. This module joins them, and — only if a local model is
actually available — asks it to phrase the result for someone who did not write
the pipeline.

Everything works with no model present. ``advise()`` returns findings and their
documentation whether or not anything can generate text; the narrator is a
bonus layer, not the mechanism. That is deliberate: the app must not acquire a
hard dependency on a language model, and a user without one should still be told
why their highlight came out the way it did.

The model is given the findings and the matching pages as its material, and told
to work only from them. It cannot invent a finding, because it does not compute
them; the worst it can do is describe a real finding badly.

Standalone usage — findings need nothing installed, narration needs a model:

    python -m modules.advisor "D:\\clips\\a_why.json"
    python -m modules.advisor "D:\\clips\\a_why.json" --llm
    python -m modules.advisor "D:\\clips\\a_why.json" --llm --ask "why so short?"
"""
from __future__ import annotations

import os
from typing import Mapping, Optional, Sequence

from modules.highlight_advice import Finding, diagnose

# docs/advisor lives beside the package, not inside it.
KNOWLEDGE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "advisor"
)

SYSTEM_PROMPT = (
    "You help someone tune a video highlight tool. You are given findings that "
    "were computed from their actual run, and documentation pages explaining "
    "those findings.\n"
    "Rules:\n"
    "1. Use ONLY the findings and pages provided. If they do not cover "
    "something, say you do not know.\n"
    "2. Never invent a number. Every figure you give must appear in the "
    "findings.\n"
    "3. Lead with the single change most likely to help, and say what it will "
    "do.\n"
    "4. Be brief and concrete. No preamble, no encouragement, no restating the "
    "question.\n"
    "5. If a finding says the detector cannot see something at all, say that "
    "settings will not fix it rather than suggesting settings."
)

# The other thing a model can usefully do with this report: read the footage
# rather than the settings. Kept as its own prompt because the two ask for
# opposite behaviour -- the advisor above must never speculate, and this is
# hired to. What makes that safe is not caution, it is separation: it says out
# loud that it is reading rather than measuring, and it cannot introduce a
# figure, so anything it suggests can be checked against the sections above it.
READING_SYSTEM_PROMPT = (
    "You are reading measurements another tool made of someone's video, and "
    "telling them what it looks like was going on. The measurements are given "
    "as plain sentences, already checked.\n"
    "Rules:\n"
    "1. Every observation must rest on something in the material given. If you "
    "want to say something it does not support, say instead that the "
    "measurements do not show it.\n"
    "2. Never state a number that is not already in the material.\n"
    "3. You may suggest what a combination of signals could mean. Mark it as "
    "your reading when you do -- 'this looks like', 'this could be' -- and "
    "never phrase a guess so it reads as something that was measured.\n"
    "4. The interesting part is how signals relate to each other across clips. "
    "Prefer one connection you can point at over a list of what each signal "
    "did.\n"
    "5. If the material says a combination is common in this video, say that it "
    "does not single any moment out, however suggestive it looks.\n"
    "6. The expression labels come from a five-class classifier that cannot "
    "tell a performed expression from a felt one. Treat them as labels on a "
    "face, and say so if you build anything on them.\n"
    "7. Be brief and concrete. No preamble, no headings, no restating the "
    "question."
)

READING_TASK = (
    "In 4 sentences or fewer: say what this video looks like it is doing, and "
    "point at the one connection between signals that a person should check "
    "first."
)

# The default ask. Deliberately tight: this lands at the top of a report whose
# findings are already listed underneath in full, so a long restatement of them
# pushes the actual evidence off the screen and earns nothing.
SUMMARY_TASK = (
    "In 3 sentences or fewer: say what shaped this highlight, and name the one "
    "change most likely to improve it. No lists, no headings, no preamble."
)

# Tokens for that summary. Three sentences do not need more, and every token is
# time the user waits.
SUMMARY_TOKENS = 200

# How much prompt a local model can be handed. llama.cpp refuses the entire call
# when the prompt overruns the context window — "Prompt exceeds n_ctx" and
# nothing generated — so this is a budget rather than a guideline: overrunning it
# costs the whole narration, not its last paragraph. Roughly four characters to
# the token against the window :func:`load_llm` asks for, leaving room for the
# system prompt and the answer.
MAX_PROMPT_CHARS = 24000

# What the report asks a llama-cpp model for. The 4096 default is smaller than
# this report's own prompt once a video has chapters and a dozen clips, and the
# failure it produces is silent at the point the user sees it.
DEFAULT_N_CTX = 8192


def knowledge_topics(knowledge_dir: str = KNOWLEDGE_DIR) -> dict:
    """Every advisor page, keyed by topic name.

    Read from disk each time rather than cached: these are meant to be edited
    while the app is running, and a stale answer is the failure mode the whole
    file-based design exists to avoid.
    """
    topics = {}
    if not os.path.isdir(knowledge_dir):
        return topics
    for name in sorted(os.listdir(knowledge_dir)):
        if not name.endswith(".md") or name == "README.md":
            continue
        path = os.path.join(knowledge_dir, name)
        try:
            with open(path, encoding="utf-8") as fh:
                topics[name[:-3]] = fh.read()
        except OSError:
            continue
    return topics


def knowledge_for(findings: Sequence[Finding],
                  knowledge_dir: str = KNOWLEDGE_DIR) -> dict:
    """Only the pages the findings actually reference."""
    topics = knowledge_topics(knowledge_dir)
    wanted = {f.topic for f in findings}
    return {name: text for name, text in topics.items() if name in wanted}


def format_findings(findings: Sequence[Finding]) -> str:
    """The findings as plain text — the debug view, and the model's input."""
    if not findings:
        return "No problems were found in this run."
    lines = []
    for i, f in enumerate(findings, start=1):
        lines.append(f"{i}. [{f.severity}] {f.title}")
        lines.append(f"   What the run shows: {f.detail}")
        lines.append(f"   Suggested change: {f.remedy}")
        lines.append(f"   See: {f.topic}")
    return "\n".join(lines)


def build_prompt(report: Mapping,
                 findings: Sequence[Finding],
                 question: Optional[str] = None,
                 knowledge_dir: str = KNOWLEDGE_DIR,
                 max_chars: int = MAX_PROMPT_CHARS) -> str:
    """What the model is given: this run, its findings, and the relevant pages.

    Assembled to a budget — see :func:`_fit`. A report with a dozen clips and a
    chapter breakdown outgrew the default context window of a local model, and
    llama.cpp answers that by refusing the call rather than by truncating, so
    the user lost the narration entirely rather than its least useful section.
    """
    totals = report.get("totals") or {}
    video = report.get("video") or {}
    settings = report.get("settings") or {}

    # (priority, text) for everything optional. The head and the findings are
    # not in here: without them there is nothing to narrate and the call is a
    # waste of the minute the user spends waiting for it.
    blocks = []
    parts = [
        "## This run",
        f"Video length: {float(video.get('duration') or 0) / 60:.0f} minutes",
        f"Highlight: {totals.get('segments', 0)} clips, "
        f"{float(totals.get('duration') or 0):.0f}s "
        f"({float(totals.get('coverage_pct') or 0):.1f}% of the source)",
        f"Settings: {', '.join(f'{k}={v}' for k, v in sorted(settings.items()))}",
        "",
        "## Findings computed from this run",
        format_findings(findings),
        "",
    ]

    # What the run actually measured about the footage, as opposed to what it
    # diagnosed about the settings. Without this the model is asked to comment
    # on a video it has been told nothing about beyond its length: it can
    # discuss weights and never the moments, which is half the question people
    # ask it. Sentences again rather than figures, for the reason below.
    try:
        from modules.highlight_prose import (clip_sections, conclude,
                                             summarise_clip)
        sections = conclude(report)
        if sections:
            lines = []
            for section in sections:
                lines.append(f"### {section['heading']}")
                lines.extend(section["lines"])
            blocks.append((1, "## What the run found in the footage\n"
                              + "\n".join(lines)))

        # Per clip: the order its marks arrived in and how unusual that
        # combination is for this video. This is the raw material for any
        # observation about how the signals relate, and it is the part the model
        # cannot reconstruct from anything else in the prompt.
        clips = []
        for entry in (report.get("segments") or [])[:12]:
            said = " ".join(summarise_clip(entry))
            if not said:
                continue
            channels = [h for h, _lines in clip_sections(entry)]
            clips.append(f"- Clip {entry.get('index', '?')} "
                         f"({entry.get('range', '')}): {said} "
                         f"[channels with something to say: "
                         f"{', '.join(channels) or 'none'}]")
        if clips:
            blocks.append((3, "## The kept clips, one line each\n"
                              + "\n".join(clips)))
    except Exception as exc:                # narration must survive this
        print(f"⚠️ Footage context skipped: {exc}")

    # Where in the video the cut came from. Worth the tokens because it answers
    # a question the findings cannot: a cut drawn entirely from one stretch of a
    # long video is a fact about the footage, and the model can only say so if
    # it is told the structure. Sentences rather than raw numbers, so the model
    # is summarising a claim that was already checked instead of doing the
    # arithmetic itself — which is where it would invent one.
    chapters = report.get("chapters") or []
    if chapters:
        try:
            from modules.highlight_prose import (describe_chapter,
                                                 summarise_chapter_run)
            lines = [summarise_chapter_run(chapters)]
            for ch in chapters:
                said = "; ".join(describe_chapter(ch))
                lines.append(f"- {ch.get('timestamp', '')} {ch.get('title', '')} "
                             f"({float(ch.get('duration') or 0):.0f}s, "
                             f"{int(ch.get('clips') or 0)} clip(s)): {said}")
            blocks.append((
                4, "## How the video divides, and where the clips came from\n"
                   + "\n".join(lines)))
        except Exception as exc:            # narration must survive this
            print(f"⚠️ Chapter context skipped: {exc}")

    docs = knowledge_for(findings, knowledge_dir)
    if docs:
        blocks.append((2, "## Documentation\n" + "\n".join(
            f"### {name}\n{text}" for name, text in docs.items())))

    return _fit("\n".join(parts), blocks,
                "## Task\n" + (question or SUMMARY_TASK), max_chars)


def _fit(head: str, blocks: Sequence, task: str, max_chars: int) -> str:
    """Head, as many blocks as fit, then the task — and a note on what was cut.

    Blocks are dropped in reverse order of what they are worth to the answer,
    lowest priority first. The task is never dropped: a model given context and
    no question answers one it invented.

    What was left out is named in the prompt. A summary written without the
    chapter breakdown must not read as a summary of a video that did not have
    one, and a model told nothing about the omission has no way to avoid that.
    """
    kept = sorted(blocks, key=lambda b: b[0])
    dropped = []
    while True:
        body = "\n\n".join(text for _priority, text in kept)
        note = ""
        if dropped:
            note = ("\n\n(Left out of this prompt for length: "
                    + ", ".join(dropped) + ". These were measured — do not "
                    "describe them as missing from the run.)")
        prompt = f"{head}\n{body}{note}\n\n{task}"
        if len(prompt) <= max_chars or not kept:
            return prompt
        lost = kept.pop()
        dropped.append(lost[1].splitlines()[0].lstrip("# ").strip())


def _generate(llm, prompt: str, system: str, max_tokens: int) -> str:
    """Call whichever text interface this object offers.

    ``LLMModule`` exposes ``query`` and takes its context-building on itself;
    the raw backends expose ``generate``. Both are accepted so the caller can
    pass whatever it already has, and ``free_chat_mode`` keeps ``query`` from
    prepending its own video context — the prompt here is complete on its own.
    """
    if hasattr(llm, "generate"):
        return llm.generate(prompt, system=system, max_tokens=max_tokens)
    if hasattr(llm, "query"):
        return llm.query(prompt, system_prompt=system, free_chat_mode=True,
                         max_tokens=max_tokens)
    raise TypeError(f"{type(llm).__name__} offers neither generate() nor query()")


def narrate(report: Mapping,
            findings: Sequence[Finding],
            *,
            llm=None,
            question: Optional[str] = None,
            knowledge_dir: str = KNOWLEDGE_DIR,
            max_tokens: int = SUMMARY_TOKENS,
            system: Optional[str] = None) -> Optional[str]:
    """Ask a local model to phrase the findings. ``None`` if none is available.

    ``llm`` is either an ``llm.llm_module.LLMModule`` or one of its backends —
    see :func:`_generate`. It is passed in rather than constructed here so that
    loading a model stays the caller's decision: this module never starts one.
    """
    if llm is None:
        return None
    prompt = build_prompt(report, findings, question, knowledge_dir)
    try:
        text = _generate(llm, prompt, system or SYSTEM_PROMPT, max_tokens)
    except Exception as exc:            # a missing model must not break the run
        print(f"⚠️ Advisor narration failed: {exc}")
        return None
    return (text or "").strip() or None


def advise(report: Mapping,
           *,
           rejected: Optional[Sequence[Sequence[float]]] = None,
           llm=None,
           question: Optional[str] = None,
           knowledge_dir: str = KNOWLEDGE_DIR) -> dict:
    """Findings, their documentation, and a narration when one is possible."""
    findings = diagnose(report, rejected=rejected)
    return {
        "findings": [f.as_dict() for f in findings],
        "text": format_findings(findings),
        "topics": sorted(knowledge_for(findings, knowledge_dir)),
        "narration": narrate(report, findings, llm=llm, question=question,
                             knowledge_dir=knowledge_dir),
    }


def summarise_report_file(json_path: str,
                          *,
                          llm,
                          question: Optional[str] = None,
                          knowledge_dir: str = KNOWLEDGE_DIR,
                          reading: bool = False) -> Optional[str]:
    """Write a short summary into a report already on disk, page and record both.

    Re-renders the HTML from the updated record rather than patching it, so the
    two cannot disagree — the same reason the record exists at all.

    Returns the summary, or ``None`` if nothing could be generated; the report
    is left untouched in that case rather than half-updated.
    """
    import json

    from modules.highlight_report import render_html

    with open(json_path, encoding="utf-8") as fh:
        report = json.load(fh)

    findings = diagnose(report)
    # Two different jobs, kept in two different fields. Tuning advice and a
    # reading of the footage answer opposite questions, and a reader who cannot
    # tell which one they are looking at will act on the wrong one.
    text = narrate(report, findings, llm=llm,
                   question=question or (READING_TASK if reading else None),
                   knowledge_dir=knowledge_dir,
                   system=READING_SYSTEM_PROMPT if reading else None)
    if not text:
        return None

    report["advice"] = [f.as_dict() for f in findings]
    report["reading" if reading else "advice_narration"] = text
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)

    html_path = json_path[:-5] + ".html" if json_path.endswith(".json") else None
    if html_path and os.path.exists(html_path):
        with open(html_path, "w", encoding="utf-8") as fh:
            fh.write(render_html(report))
    return text


def load_llm(backend: str = "ollama", model: str = "llama3",
             mmproj: Optional[str] = None, n_ctx: int = DEFAULT_N_CTX):
    """Load a local model for narration, or return ``None`` with a reason.

    Never called automatically: narration costs seconds to minutes, and paying
    that on every render — for prose that repeats findings already on the page —
    would be a poor trade.

    ``model`` is an Ollama tag or, for ``llama-cpp``, the path to a ``.gguf``.
    The two go to different constructor arguments — ``LLMModule`` takes a name
    as ``model`` and a file as ``model_path``, and passing a path as the name
    left the backend with an empty path and a "GGUF model not found:" that named
    no file. The model picker has offered a .gguf path since it was written, so
    every attempt to use one from here failed that way.
    """
    try:
        from llm.llm_module import LLMModule
    except Exception as exc:
        print(f"⚠️ No LLM stack available: {exc}")
        return None
    try:
        if backend == "llama-cpp":
            llm = LLMModule(backend=backend, model_path=model,
                            mmproj_path=mmproj, n_ctx=n_ctx)
        else:
            llm = LLMModule(backend=backend, model=model)
        llm.load()
        return llm
    except Exception as exc:
        print(f"⚠️ Could not load {backend}/{model}: {exc}")
        return None


def _main(argv=None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        prog="python -m modules.advisor",
        description="Say why a highlight came out the way it did.")
    parser.add_argument("report", help="the *_why.json written beside a cut")
    parser.add_argument("--llm", action="store_true",
                        help="also phrase the findings with a local model")
    parser.add_argument("--backend", default="ollama",
                        choices=("ollama", "llama-cpp"))
    parser.add_argument("--model", default="llama3")
    parser.add_argument("--ask", metavar="QUESTION",
                        help="ask something specific instead of the summary")
    args = parser.parse_args(argv)

    with open(args.report, encoding="utf-8") as fh:
        report = json.load(fh)

    findings = diagnose(report)
    print(format_findings(findings))

    if not (args.llm or args.ask):
        return 0
    llm = load_llm(args.backend, args.model)
    if llm is None:
        return 1
    text = narrate(report, findings, llm=llm, question=args.ask)
    if text:
        print("\n" + "-" * 70 + "\n" + text)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
