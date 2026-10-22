"""Tests for re-timing segments Whisper stretched over silence.

No Whisper here: the aligner is a stub returning word timings shaped like the
real one's, where the first word soaks up the quiet stretch before it. What's
pinned down: only stretched segments are touched, the text never changes, the
start lands at the first word, an unsure alignment keeps Whisper's times, and
the next segment only ever has its start moved later.
"""
from __future__ import annotations

import pytest

from modules.audio import segment_timing as st


def _seg(start, end, text):
    return {"start": float(start), "end": float(end), "text": text}


def _words(start, pairs, p=0.9):
    """Words from (end, text) pairs, each starting where the last ended."""
    out, at = [], start
    for end, text in pairs:
        out.append((at, end, text, p))
        at = end
    return out


# A ten-word question said in the last three seconds of a 26.6 s segment.
QUESTION = _seg(272.8, 299.4, "So, do you know why you were put in there?")
QUESTION_WORDS = _words(272.8, [(296.68, " So"), (296.88, ","), (297.66, " do"),
                                (298.04, " you"), (298.08, " know"), (298.20, " why"),
                                (298.36, " you"), (298.44, " were"), (298.54, " put"),
                                (298.70, " in"), (299.20, " there"), (299.38, "?")])


def _aligner(*per_text, calls=None):
    def align(start, end, texts):
        if calls is not None:
            calls.append((start, end, list(texts)))
        return list(per_text)[:len(texts)] + [[] for _ in range(len(texts) - len(per_text))]
    return align


class TestIsStretched:
    @pytest.mark.parametrize("seg, expected", [
        (QUESTION, True),
        (_seg(0, 4, "a short line"), False),
        (_seg(0, 9, "one two three four five six seven eight nine ten"), False),
        (_seg(0, 12, "one two"), True),
        (_seg(0, 30, ""), False),
    ])
    def test_far_longer_than_its_words(self, seg, expected):
        assert st.is_stretched(seg) is expected


class TestBoundsFromWords:
    def test_the_silence_soaked_up_by_the_first_word_is_cut(self):
        start, end = st.bounds_from_words(QUESTION_WORDS)
        assert 296.1 < start < 296.68
        # The last word is capped the same way, from the end.
        assert 298.9 < end <= 299.2

    def test_a_lone_word_keeps_its_end(self):
        start, end = st.bounds_from_words([(0.0, 11.7, " next", 0.9)])
        assert end == pytest.approx(11.7) and start < end

    def test_punctuation_only_is_no_speech(self):
        assert st.bounds_from_words([(0.0, 1.0, ",", 0.9), (1.0, 2.0, "?", 0.9)]) is None
        assert st.bounds_from_words([]) is None


class TestRetime:
    def test_a_stretched_segment_starts_when_it_was_said(self):
        later = _seg(310.0, 312.0, "No.")                   # not following on
        out = st.retime([QUESTION, later], _aligner(QUESTION_WORDS), log_fn=lambda m: None)
        assert 296.1 < out[0]["start"] < 296.7
        assert 298.9 < out[0]["end"] <= 299.2
        assert out[0]["text"] == QUESTION["text"]
        assert out[1] is later

    def test_only_stretched_segments_are_aligned(self):
        calls = []
        st.retime([_seg(0, 2, "fine as is"), _seg(40, 41, "x"), QUESTION],
                  _aligner(QUESTION_WORDS, calls=calls), log_fn=lambda m: None)
        assert [texts for _, _, texts in calls] == [[QUESTION["text"]]]

    def test_alone_it_is_used_as_it_always_was(self):
        # No confidence gate when there is no next segment to weigh it against:
        # this is the behaviour the joint alignment only ever improves on.
        unsure = [(s, e, w, 0.05) for s, e, w, _ in QUESTION_WORDS]
        out = st.retime([QUESTION], _aligner(unsure), log_fn=lambda m: None)
        assert 296.1 < out[0]["start"] < 296.7

    @pytest.mark.parametrize("align", [
        lambda start, end, texts: (_ for _ in ()).throw(RuntimeError("no GPU")),
        lambda start, end, texts: [[] for _ in texts],
    ])
    def test_a_failure_keeps_whispers_times(self, align):
        assert st.retime([QUESTION], align, log_fn=lambda m: None) == [QUESTION]


class TestWithTheNextSegment:
    """Whisper ended the segment before its line was said, and started the
    next one late by as much: the line belongs after Whisper's end."""

    LINE = _seg(673.0, 696.72, "Wait for it, everyone.")
    NEXT = _seg(696.72, 701.52, "Then the next part follows here.")
    LINE_WORDS = _words(673.0, [(698.12, " Wait"), (698.20, " for"), (698.38, " it"),
                                (698.60, ","), (698.78, " everyone"), (699.22, ".")])

    def test_the_next_segment_is_aligned_with_it(self):
        calls = []
        st.retime([self.LINE, self.NEXT], _aligner(self.LINE_WORDS, calls=calls),
                  log_fn=lambda m: None)
        assert calls == [(673.0, 701.52, [self.LINE["text"], self.NEXT["text"]])]

    def test_the_line_may_end_after_whispers_end(self):
        out = st.retime([self.LINE, self.NEXT], _aligner(self.LINE_WORDS),
                        log_fn=lambda m: None)
        assert 697.5 < out[0]["start"] < 698.12
        assert self.LINE["end"] < 698.5 < out[0]["end"] <= 699.22
        # The next one starts no earlier than the line now ends; nothing else moves.
        assert out[1]["start"] == pytest.approx(out[0]["end"])
        assert out[1]["end"] == self.NEXT["end"] and out[1]["text"] == self.NEXT["text"]

    ALONE_WORDS = _words(673.0, [(690.9, " Wait"), (691.4, " for"), (691.9, " it"),
                                 (692.5, " everyone")], p=0.1)

    def _by_count(self, joint, alone):
        return lambda start, end, texts: [joint if len(texts) == 2 else alone] + [[]] * (len(texts) - 1)

    def test_an_unsure_joint_alignment_falls_back_to_alone(self):
        unsure = [(s, e, w, 0.1) for s, e, w, _ in self.LINE_WORDS]
        out = st.retime([self.LINE, self.NEXT], self._by_count(unsure, self.ALONE_WORDS),
                        log_fn=lambda m: None)
        assert 689 < out[0]["start"] < 691 and out[1] is self.NEXT

    def test_a_joint_alignment_that_squeezes_the_next_line_falls_back(self):
        # Seven words left 0.3 s: the line took the next one's speech.
        nxt = _seg(696.72, 699.5, "one two three four five six seven")
        out = st.retime([self.LINE, nxt], self._by_count(self.LINE_WORDS, self.ALONE_WORDS),
                        log_fn=lambda m: None)
        assert 689 < out[0]["start"] < 691 and out[1] is nxt

    def test_the_next_segment_only_ever_moves_later(self):
        early = _words(673.0, [(690.0, " Wait"), (690.2, " for"), (690.4, " it"),
                               (690.8, " everyone")])
        out = st.retime([self.LINE, self.NEXT], _aligner(early), log_fn=lambda m: None)
        assert out[1] is self.NEXT

    def test_a_segment_far_after_is_not_joined(self):
        calls = []
        far = _seg(705.0, 707.0, "Later.")
        st.retime([self.LINE, far], _aligner(self.LINE_WORDS, calls=calls),
                  log_fn=lambda m: None)
        assert calls == [(673.0, 696.72, [self.LINE["text"]])]

    def test_never_past_one_window(self):
        calls = []
        long_next = _seg(696.72, 704.0, "Then the next part follows here.")
        line = _seg(673.5 - 1.0, 696.72, self.LINE["text"])
        st.retime([line, long_next], _aligner(self.LINE_WORDS, calls=calls),
                  log_fn=lambda m: None)
        assert calls[0][1] == 696.72          # 704 - 672.5 is over 30 s
