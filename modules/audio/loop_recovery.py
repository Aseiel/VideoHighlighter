"""Get back the speech a Whisper repetition loop swallowed.

Whisper decodes 30 seconds at a time. When a window *opens* on a sound it
cannot make out (a gurgle, a laugh, a sting of music) it can lock onto one
token and repeat it until the window's token budget is spent: a single segment
reading "Um, um, um, ..." two hundred times. ``transcript_cleanup`` collapses
that to one "Um." - and everything that was actually said in the rest of the
window is gone with it, because the loop used the window up. The transcript
then shows a silent half minute in the middle of a conversation.

Measured 2026-11-13 on a 20-minute cartoon episode with the 'small' model: a
loop starting at 6:57 took the window to 7:27, eleven lines of dialogue. The
same audio decoded again from 1.5 seconds later came back as speech, and from
a few seconds later still, every line of it. The loop is a property of where
the window started, not of the audio after it.

So this looks for exactly that shape - a looped segment followed by a hole -
and decodes the hole again from a little further in, trying a few starts and
keeping the first that does not loop straight away. It is not the general
"re-decode whatever looks wrong" that ``whisper_openvino`` measured and
rejected: that re-decoded most of a noisy video for nothing. A loop with a
hole behind it is rare (once in this episode) and the retry is a few seconds
of audio.

Nothing here knows about a model. The caller passes ``redecode(start, end)``,
which returns segments timed from the start of the same audio the input
segments are timed against.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

from modules.audio.transcript_cleanup import collapse_repeats

# A hole shorter than this holds a line at most, and a looped segment that runs
# straight into the next line has swallowed nothing.
MIN_HOLE_SECONDS = 4.0

# How far into the hole each retry starts. The loop comes from the sound the
# window opened on, so the retries step past it: short first, because every
# second skipped is a second that is not transcribed.
RETRY_OFFSETS = (1.5, 3.0, 5.0, 8.0)

# Less than this left to decode is not worth a call.
MIN_RETRY_SECONDS = 2.0

# A recovered stretch can loop again further in; one more look, then stop.
MAX_PASSES = 2

Segment = Dict
Redecode = Callable[[float, float], List[Segment]]


def is_loop(segment: Segment) -> bool:
    """Whether this segment's text is a repetition loop: collapsing its
    back-to-back repeats (``transcript_cleanup.collapse_repeats``) changes it."""
    text = str(segment.get("text") or "").strip()
    return bool(text) and collapse_repeats(text) != text


def swallowed_stretches(segments: List[Segment], total_seconds: float,
                        min_hole: float = MIN_HOLE_SECONDS
                        ) -> List[Tuple[int, int, float, float]]:
    """``(first, after, start, end)`` for every loop with a hole behind it.

    ``first`` is the index of the looped segment, ``after`` the index of the
    next segment that is not a loop (``len(segments)`` when there is none),
    and ``start``/``end`` the stretch of audio between them. Timed from where
    the loop *started*: a looped segment's own end is where the decoder gave
    up, which says nothing about the audio.
    """
    found = []
    i = 0
    while i < len(segments):
        if not is_loop(segments[i]):
            i += 1
            continue
        j = i + 1
        while j < len(segments) and is_loop(segments[j]):
            j += 1
        start = float(segments[i].get("start", 0.0))
        end = float(segments[j].get("start", total_seconds)) if j < len(segments) \
            else float(total_seconds)
        if end - start >= min_hole:
            found.append((i, j, start, end))
        i = j
    return found


def _retry(start: float, end: float, redecode: Redecode) -> Optional[List[Segment]]:
    """The hole decoded again from the first start that does not loop at once,
    or None when every start did (or there was nothing there)."""
    for offset in RETRY_OFFSETS:
        begin = start + offset
        if end - begin < MIN_RETRY_SECONDS:
            break
        got = [s for s in (redecode(begin, end) or [])
               if str(s.get("text") or "").strip()]
        if got and not is_loop(got[0]):
            # Whisper pads its last timestamp to the end of what it was given;
            # nothing recovered may run into the line that follows the hole.
            kept = []
            for s in got:
                s_start = float(s.get("start", begin))
                if s_start >= end:
                    continue
                new = dict(s)
                new["start"] = max(s_start, begin)
                new["end"] = min(max(float(s.get("end", s_start)), new["start"]), end)
                kept.append(new)
            if kept:
                return kept
    return None


def recover(segments: List[Segment], redecode: Redecode, total_seconds: float,
            log_fn=None) -> List[Segment]:
    """``segments`` with the speech behind each loop decoded again.

    The looped segments themselves stay where they were (the cleanup that
    follows collapses or drops them); what is recovered goes in after them.
    A stretch that loops on every retry is left as it was. Returns a new list.
    """
    out = list(segments or [])
    # Loops already tried, by where they start: what was recovered goes in
    # after the loop, so the loop is still there on the next pass with a
    # (shorter) hole behind it, and must not be decoded a second time.
    tried = set()
    for _ in range(MAX_PASSES):
        stretches = [s for s in swallowed_stretches(out, total_seconds)
                     if s[2] not in tried]
        if not stretches:
            break
        changed = False
        # Back to front, so the indices of the earlier ones stay true.
        for first, after, start, end in reversed(stretches):
            tried.add(start)
            got = _retry(start, end, redecode)
            if not got:
                continue
            # The loop keeps its start and gives up its claim on the hole.
            head = []
            for seg in out[first:after]:
                seg = dict(seg)
                seg["end"] = min(float(seg.get("end", start)),
                                 float(got[0]["start"]))
                seg["end"] = max(seg["end"], float(seg.get("start", start)))
                head.append(seg)
            out[first:after] = head + got
            changed = True
            if log_fn:
                log_fn(f"🔁 Whisper looped at {_clock(start)}; decoded "
                       f"{_clock(start)}-{_clock(end)} again and recovered "
                       f"{len(got)} line(s)")
        if not changed:
            break
    return out


def _clock(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"
