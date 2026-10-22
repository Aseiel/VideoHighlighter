"""Take Whisper's repetition loops out of a transcript.

Whisper decodes 30 seconds at a time, and on audio it cannot make out (noise,
music, breathing, a pause) it can lock onto one phrase and repeat it: one
segment holding "we can go, we can go, ..." fifty times over, or a run of
one-second segments that all say the same short line. The decoder settings
(modules/audio/transcript.py) make that rarer; this catches what still gets
through, on new transcripts and on ones already in the analysis cache.

Three rules, all about the text's shape and nothing about its subject:

* **A phrase repeated back to back inside a segment** collapses to one
  occurrence: four or more times for a phrase of several words, six or more for
  a single word. Fewer is left alone; people do say "no, no, no". The segment
  is then shortened to about the time it takes to say what is left, since the
  loop had stretched it over the whole window.
* **A run of three or more consecutive segments with the same text**, each
  starting within a minute of the one before, keeps only the first. Two in a
  row stay ("Yeah." "Yeah." happens), and so do lines further apart: a loop
  repeats within a window, and the ghost of a quiet stretch once per window,
  30 seconds on.
* **A segment Whisper itself thought was not speech**, when its own numbers
  say so (no-speech probability high, confidence low), is dropped. Only
  segments that carry those numbers are judged; cached ones usually do not.

Every function here is idempotent, so cleaning an already clean transcript
changes nothing.
"""
from __future__ import annotations

import re
from typing import Dict, List

# Back-to-back repeats before a phrase counts as a loop.
MIN_PHRASE_REPEATS = 4          # phrases of two words or more
MIN_WORD_REPEATS = 6            # a single word
MAX_PHRASE_WORDS = 12

MIN_SEGMENT_RUN = 3             # identical consecutive segments
MAX_RUN_GAP = 60.0              # seconds from one to the next, start to start

# Whisper's own judgement of a segment. It already skips a window when
# no_speech_prob > 0.6 *and* avg_logprob < -1.0; a short ghost on a quiet
# stretch (a stock phrase from Whisper's training data) tends to pass the
# second test, so this asks less.
NO_SPEECH_PROB = 0.6
NO_SPEECH_LOGPROB = -0.7

# Seconds a collapsed segment keeps: about what it takes to say it.
_SECONDS_PER_WORD = 0.45
_MIN_SECONDS = 1.5

_WORD = re.compile(r"\w+", re.UNICODE)


def _norm(token: str) -> str:
    return "".join(_WORD.findall(token.lower()))


def collapse_repeats(text: str) -> str:
    """``text`` with every looped phrase said once.

    Tokens are compared without case or punctuation, so "we can go, We can
    go. we can go" is three of the same phrase. A broken-off repeat at the end
    of a loop ("... we can go, we") goes with the loop.
    """
    tokens = text.split()
    keys = [_norm(t) for t in tokens]
    out: List[str] = []
    i = 0
    while i < len(tokens):
        best_n, best_k = 0, 0
        for n in range(1, min(MAX_PHRASE_WORDS, (len(tokens) - i) // 2) + 1):
            phrase = keys[i:i + n]
            if not all(phrase):
                continue
            k = 1
            while keys[i + k * n:i + (k + 1) * n] == phrase:
                k += 1
            need = MIN_WORD_REPEATS if n == 1 else MIN_PHRASE_REPEATS
            if k >= need and n * k > best_n * best_k:
                best_n, best_k = n, k
        if not best_n:
            out.append(tokens[i])
            i += 1
            continue
        out.extend(tokens[i:i + best_n])
        i += best_n * best_k
        # The loop's broken-off tail: the first words of the phrase once more.
        tail = 0
        while (tail < best_n and i + tail < len(tokens)
               and keys[i + tail] == keys[i - best_n + tail]):
            tail += 1
        if 0 < tail < best_n and i + tail == len(tokens):
            i += tail
    collapsed = " ".join(out)
    if collapsed != text.strip() and collapsed[-1:] == ",":
        collapsed = collapsed[:-1] + "."
    return collapsed


def _spoken_seconds(text: str) -> float:
    return max(_MIN_SECONDS, _SECONDS_PER_WORD * len(_WORD.findall(text)))


def _looks_like_no_speech(seg: Dict) -> bool:
    nsp, lp = seg.get("no_speech_prob"), seg.get("avg_logprob")
    if nsp is None or lp is None:
        return False
    return float(nsp) > NO_SPEECH_PROB and float(lp) < NO_SPEECH_LOGPROB


def clean_segments(segments: List[Dict]) -> List[Dict]:
    """The segments without Whisper's loops (see the module docstring).

    Returns new dicts; the input is not modified. Every other key a segment
    carries (speaker labels, Whisper's numbers) is kept.
    """
    cleaned: List[Dict] = []
    for seg in segments or []:
        text = str(seg.get("text") or "").strip()
        if not text or _looks_like_no_speech(seg):
            continue
        new = dict(seg)
        collapsed = collapse_repeats(text)
        if collapsed != text:
            new["text"] = collapsed
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", start))
            new["end"] = min(end, start + _spoken_seconds(collapsed))
        cleaned.append(new)

    out: List[Dict] = []
    i = 0
    while i < len(cleaned):
        key = _norm(cleaned[i]["text"])
        j = i + 1
        while (j < len(cleaned) and key and _norm(cleaned[j]["text"]) == key
               and float(cleaned[j].get("start", 0.0))
               - float(cleaned[j - 1].get("start", 0.0)) <= MAX_RUN_GAP):
            j += 1
        out.extend(cleaned[i:j] if j - i < MIN_SEGMENT_RUN else cleaned[i:i + 1])
        i = j
    return out
