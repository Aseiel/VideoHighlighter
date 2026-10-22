"""Give a stretched Whisper segment the times its words were actually said.

Without word timestamps, Whisper starts a segment where the previous one
ended, so a line spoken after a quiet stretch carries that whole stretch with
it. One ten-word question came out as 272.8 to 299.4 s and showed on screen
23 seconds before anyone said it.

Asking Whisper for word timestamps throughout fixes that but changes how it
steps through the audio, and with it the words: measured on the same ten
minutes, it garbled lines the plain decode had right and added empty
segments. So the decode stays as it is, and only the segments that are
obviously stretched (far longer than their words take to say) are re-timed,
with Whisper's aligner (``whisper.timing.find_alignment``) and their own
text, which is never changed.

**With the next segment.** Whisper can also end a stretched segment before
its line is said, and start the next one late by the same amount. Aligned
alone, the words then have to fit before that wrong end, and the aligner
guesses. So a stretched segment is aligned together with the segment after it,
over the audio up to that one's end (both inside one 30-second window), and
the words find their place across the two.

**Only when that is safe.** The joint times are used when the aligner's mean
confidence in this segment's words is at least ``MIN_CONFIDENCE`` *and* the
next segment is left time to say its own words; across one film, the joint
alignment otherwise sometimes pulled the next line's speech into this one.
Failing either, the segment is aligned alone, as it was before the joint
alignment existed. Of the next segment, only its start moves, and only later,
when this line now runs into it. Its text and end never change.

The aligner has to begin at the start of the audio it is given, so the first
word soaks up the silence before it; its *end* is right. As in Whisper's own
word timestamps, a first or last word longer than twice the median word is
cut back to that, from the side that touches the silence.

Any failure leaves the segments as Whisper had them.
"""
from __future__ import annotations

import re
import statistics
from typing import Callable, List, Optional, Sequence, Tuple

# A segment this much longer than its words is re-timed.
SECONDS_PER_WORD = 1.0
MIN_STRETCHED = 6.0
MAX_WORD_SECONDS = 0.7          # Whisper's cap on the median word

# The aligner's mean word probability for the segment, below which its times
# are not trusted. A line it placed well scored 0.7-0.85; squeezed into the
# wrong stretch of audio, the same line scored 0.15.
MIN_CONFIDENCE = 0.4

# The next segment joins the alignment when it follows on (Whisper puts one
# right after the other) and both fit in the one window the aligner reads.
MAX_FOLLOW_GAP = 1.0
WINDOW_SECONDS = 30.0
# What the next segment must keep after a joint alignment: fast speech is
# three to four words a second.
NEXT_SECONDS_PER_WORD = 0.25

_WORD = re.compile(r"\w+", re.UNICODE)

Word = Tuple[float, float, str, float]   # start, end, text, probability (absolute seconds)
Aligner = Callable[[float, float, List[str]], List[Sequence[Word]]]


def is_stretched(seg: dict) -> bool:
    """Far longer than its words take to say."""
    words = len(_WORD.findall(str(seg.get("text") or "")))
    if not words:
        return False
    duration = float(seg.get("end", 0.0)) - float(seg.get("start", 0.0))
    return duration > max(MIN_STRETCHED, SECONDS_PER_WORD * words)


def _spoken(words: Sequence[Word]) -> List[Word]:
    return [w for w in words if _WORD.search(w[2] or "")]


def confidence(words: Sequence[Word]) -> float:
    """The aligner's mean probability over the words (punctuation aside)."""
    spoken = _spoken(words)
    return sum(w[3] for w in spoken) / len(spoken) if spoken else 0.0


def bounds_from_words(words: Sequence[Word]) -> Optional[Tuple[float, float]]:
    """``(start, end)`` of the speech in ``words``, or None when there is none."""
    spoken = _spoken(words)
    if not spoken:
        return None
    durations = [w[1] - w[0] for w in spoken if w[1] > w[0]]
    median = min(MAX_WORD_SECONDS, statistics.median(durations)) if durations else 0.0
    longest = 2 * median
    first_start, first_end = spoken[0][0], spoken[0][1]
    last_start, last_end = spoken[-1][0], spoken[-1][1]
    if longest > 0:
        first_start = max(first_start, first_end - longest)
        # A lone word soaked up the silence before it, not after: its end stands.
        if len(spoken) > 1:
            last_end = min(last_end, last_start + longest)
    return first_start, max(last_end, first_start)


def _follows(seg: dict, nxt: Optional[dict]) -> bool:
    return (nxt is not None
            and float(nxt["start"]) - float(seg["end"]) <= MAX_FOLLOW_GAP
            and float(nxt["end"]) - float(seg["start"]) <= WINDOW_SECONDS)


def _joint_bounds(seg: dict, nxt: dict, align: Aligner) -> Optional[Tuple[float, float]]:
    """``seg``'s times from aligning it together with ``nxt``, or None when
    that result is not to be trusted (see the module docstring)."""
    words = align(float(seg["start"]), float(nxt["end"]), [str(seg["text"]), str(nxt["text"])])[0]
    bounds = bounds_from_words(words)
    if bounds is None or confidence(words) < MIN_CONFIDENCE:
        return None
    start, end = bounds[0], min(bounds[1], float(nxt["end"]))
    room = float(nxt["end"]) - max(end, float(nxt["start"]))
    if room < NEXT_SECONDS_PER_WORD * len(_WORD.findall(str(nxt["text"]))):
        return None
    return start, end


def retime(segments: List[dict], align: Aligner,
           log_fn: Callable[[str], None] = print) -> List[dict]:
    """``segments`` with every stretched one re-timed (see the module docstring).

    ``align(start, end, texts)`` returns, for each of ``texts`` in order, its
    word timings over the audio from ``start`` to ``end``. Segments that change
    are new dicts; the rest are the same objects.
    """
    out = list(segments)
    moved = 0
    for i, seg in enumerate(out):
        if not is_stretched(seg):
            continue
        nxt = out[i + 1] if i + 1 < len(out) else None
        start = float(seg["start"])
        try:
            bounds = _joint_bounds(seg, nxt, align) if _follows(seg, nxt) else None
            joined = bounds is not None
            if not joined:
                bounds = bounds_from_words(align(start, float(seg["end"]), [str(seg["text"])])[0])
        except Exception as e:  # noqa: BLE001 - timing is not worth a transcript
            print(f"segment_timing: could not align {str(seg.get('text', ''))[:40]!r}: "
                  f"{type(e).__name__}: {e}")
            continue
        if bounds is None:
            continue
        out[i] = dict(seg, start=bounds[0], end=bounds[1])
        moved += 1
        if joined and float(nxt["start"]) < bounds[1]:
            out[i + 1] = dict(nxt, start=bounds[1])
    if moved:
        log_fn(f"⏱️ Re-timed {moved} segment(s) that Whisper had stretched over silence")
    return out


def whisper_aligner(model, audio, language: str) -> Aligner:
    """An ``align`` for :func:`retime`: Whisper's own aligner on ``model``,
    over ``audio`` (the chunk, as ``whisper.load_audio`` returns it)."""
    import torch
    from whisper.audio import N_FRAMES, SAMPLE_RATE, log_mel_spectrogram, pad_or_trim
    from whisper.timing import find_alignment
    from whisper.tokenizer import get_tokenizer

    tokenizer = get_tokenizer(model.is_multilingual, num_languages=model.num_languages,
                              language=language, task="transcribe")
    hop = SAMPLE_RATE * 30 // N_FRAMES           # samples per mel frame
    dtype = next(model.parameters()).dtype

    def align(start: float, end: float, texts: List[str]) -> List[Sequence[Word]]:
        piece = audio[int(start * SAMPLE_RATE):int(end * SAMPLE_RATE)]
        # One window is all the aligner reads.
        if len(piece) < hop or len(piece) > N_FRAMES * hop:
            return [[] for _ in texts]
        mel = log_mel_spectrogram(torch.from_numpy(piece), model.dims.n_mels)
        mel = pad_or_trim(mel, N_FRAMES).to(model.device, dtype)
        per_text = [tokenizer.encode(" " + t.strip()) for t in texts]
        timings = find_alignment(model, tokenizer, [t for ts in per_text for t in ts],
                                 mel, len(piece) // hop)
        # Hand each text back its own words, by how many tokens it brought.
        out: List[List[Word]] = [[] for _ in texts]
        which, used = 0, 0
        for t in timings:
            while which < len(texts) - 1 and used >= len(per_text[which]):
                which, used = which + 1, 0
            out[which].append((start + float(t.start), start + float(t.end), t.word,
                               float(t.probability)))
            used += len(t.tokens)
        return out

    return align
