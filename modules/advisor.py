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
                 knowledge_dir: str = KNOWLEDGE_DIR) -> str:
    """What the model is given: this run, its findings, and the relevant pages."""
    totals = report.get("totals") or {}
    video = report.get("video") or {}
    settings = report.get("settings") or {}

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
        "## Documentation",
    ]
    for name, text in knowledge_for(findings, knowledge_dir).items():
        parts.append(f"### {name}\n{text}")

    parts.append("")
    parts.append("## Task")
    parts.append(question or SUMMARY_TASK)
    return "\n".join(parts)


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
            max_tokens: int = SUMMARY_TOKENS) -> Optional[str]:
    """Ask a local model to phrase the findings. ``None`` if none is available.

    ``llm`` is either an ``llm.llm_module.LLMModule`` or one of its backends —
    see :func:`_generate`. It is passed in rather than constructed here so that
    loading a model stays the caller's decision: this module never starts one.
    """
    if llm is None:
        return None
    prompt = build_prompt(report, findings, question, knowledge_dir)
    try:
        text = _generate(llm, prompt, SYSTEM_PROMPT, max_tokens)
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
                          knowledge_dir: str = KNOWLEDGE_DIR) -> Optional[str]:
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
    text = narrate(report, findings, llm=llm, question=question,
                   knowledge_dir=knowledge_dir)
    if not text:
        return None

    report["advice"] = [f.as_dict() for f in findings]
    report["advice_narration"] = text
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)

    html_path = json_path[:-5] + ".html" if json_path.endswith(".json") else None
    if html_path and os.path.exists(html_path):
        with open(html_path, "w", encoding="utf-8") as fh:
            fh.write(render_html(report))
    return text


def load_llm(backend: str = "ollama", model: str = "llama3"):
    """Load a local model for narration, or return ``None`` with a reason.

    Never called automatically: narration costs seconds to minutes, and paying
    that on every render — for prose that repeats findings already on the page —
    would be a poor trade.
    """
    try:
        from llm.llm_module import LLMModule
    except Exception as exc:
        print(f"⚠️ No LLM stack available: {exc}")
        return None
    try:
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
