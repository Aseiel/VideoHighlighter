"""Draft release notes from the commits since the previous release tag.

    python tools/release_notes.py                     # HEAD vs. the last tag
    python tools/release_notes.py --since 0.9.0-Pro --until 0.11.0
    python tools/release_notes.py --out release-notes.md

The build workflow runs this and puts the result at the top of the draft GitHub
Release, which is where the notes live: nobody has to remember to write them,
and the draft has to be opened to be published anyway, so that is where they get
edited into shape.

Commit subjects are already written as "type(scope): what changed", so they are
grouped by type rather than rewritten. Work a customer never sees — docs, tests,
CI, release plumbing — is left out, and the notes say how many commits that was,
so a short list reads as "little changed", never as "something got dropped".

A squash merge titled after its branch ("Claude/amazing ptolemy yf1m4v (#22)")
says nothing, but GitHub lists the commits it squashed in its body as
"* type(scope): ..." lines, so those are used in its place.

"Previous release" is the nearest version tag reachable from ``--until``'s
parent, found by ancestry rather than by name. Tag names here have not been
consistent (``0.9.0-Pro``, ``0.11.0``), and a clone that also fetches the other
edition holds its tags too — unrelated history, so never reachable, so never
picked. Only tags starting with a digit count: ``packs-torch-2.7.1`` marks
published PyTorch packs, not a release.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

# Section per commit type, in the order they are printed.
SECTIONS = (
    ("feat", "New"),
    ("perf", "Faster"),
    ("fix", "Fixed"),
)

# Types that never reach a user of the built app.
INTERNAL_TYPES = frozenset(
    {"docs", "test", "tests", "ci", "chore", "build", "refactor", "style", "release",
     "tools"})

# Scopes that are internal whatever the type: "fix(ci)" repairs the pipeline,
# not the app.
INTERNAL_SCOPES = frozenset({"ci", "release", "tests", "test", "dev", "tools"})

_SUBJECT = re.compile(r"^(?P<type>[a-z]+)(?:\((?P<scope>[^)]*)\))?!?:\s*(?P<text>.+)$")
_BULLET = re.compile(r"^\*\s+(?P<line>.+)$")

_FIELD = "\x1f"
_RECORD = "\x1e"


@dataclass(frozen=True)
class Commit:
    sha: str
    subject: str
    body: str = ""


def subjects(commit: Commit) -> list[str]:
    """What one commit says it changed: its subject, or — for a squash merge
    titled after its branch — the conventional subjects listed in its body."""
    if _SUBJECT.match(commit.subject.strip()):
        return [commit.subject]
    listed = []
    for line in commit.body.splitlines():
        m = _BULLET.match(line.strip())
        if m and _SUBJECT.match(m["line"].strip()):
            listed.append(m["line"].strip())
    return listed or [commit.subject]


def classify(subject: str) -> tuple[Optional[str], str, str]:
    """(section title or None when internal, scope, text) for one subject.

    A subject without a type prefix still ships something, so it lands in
    "Other" rather than being hidden.
    """
    m = _SUBJECT.match(subject.strip())
    if not m:
        return "Other", "", subject.strip()
    kind, scope, text = m["type"], (m["scope"] or "").strip(), m["text"].strip()
    if kind in INTERNAL_TYPES or scope in INTERNAL_SCOPES:
        return None, scope, text
    return dict(SECTIONS).get(kind, "Other"), scope, text


def render(commits: Sequence[Commit], since: Optional[str]) -> str:
    grouped: dict[str, list[str]] = {}
    internal = 0
    # A PR branched off another unmerged one lists that PR's commits in its
    # own squash body too, so the same subject can arrive two or three times.
    seen = set()
    for c in commits:
        for subject in subjects(c):
            if subject.strip() in seen:
                continue
            seen.add(subject.strip())
            section, scope, text = classify(subject)
            if section is None:
                internal += 1
                continue
            line = text[:1].upper() + text[1:]
            if scope:
                line += f" ({scope})"
            grouped.setdefault(section, []).append(f"- {line}")

    heading = f"## What changed since {since}" if since else "## What changed"
    out = [heading, ""]
    order = [title for _, title in SECTIONS] + ["Other"]
    for title in order:
        if title in grouped:
            out += [f"### {title}", *grouped[title], ""]
    if not grouped:
        out += ["No user-facing changes.", ""]
    if internal:
        noun = "commit" if internal == 1 else "commits"
        out += [f"_{internal} internal {noun} (docs, tests, CI, release) not listed._", ""]
    return "\n".join(out)


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True,
                          text=True, encoding="utf-8").stdout


def previous_tag(until: str) -> Optional[str]:
    """Nearest version tag reachable from ``until``'s first parent, or None.

    Starting from the parent means a tagged ``until`` is compared against the
    release before it, and an untagged one against the latest release.
    """
    try:
        return _git("describe", "--tags", "--abbrev=0", "--match", "[0-9]*",
                    f"{until}^").strip() or None
    except subprocess.CalledProcessError:
        return None  # no tags yet, or a root commit


def collect(since: Optional[str], until: str) -> list[Commit]:
    span = f"{since}..{until}" if since else until
    raw = _git("log", "--no-merges", "--reverse",
               f"--format=%H{_FIELD}%s{_FIELD}%b{_RECORD}", span)
    commits = []
    for record in raw.split(_RECORD):
        record = record.strip()
        if record:
            sha, subject, body = record.split(_FIELD, 2)
            commits.append(Commit(sha, subject, body))
    return commits


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--since", help="previous release ref (default: nearest tag)")
    parser.add_argument("--until", default="HEAD", help="ref being released")
    parser.add_argument("--out", help="write here instead of stdout")
    args = parser.parse_args(list(argv) if argv is not None else None)

    since = args.since or previous_tag(args.until)
    notes = render(collect(since, args.until), since)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(notes)
    else:
        sys.stdout.reconfigure(encoding="utf-8")
        print(notes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
