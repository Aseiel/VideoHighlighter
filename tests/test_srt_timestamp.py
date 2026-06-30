"""
Characterization tests for `transcript_srt.format_timestamp_srt`.

SRT timestamps must follow the strict format `HH:MM:SS,mmm` (note the comma,
not a dot — VLC, ffmpeg subtitle muxer, and most players are strict here).
This module-level function is called by every subtitle write path, so it is
the right place to pin behaviour before any translation-stack refactor.

`transcript_srt.py` does a top-level `import whisper`, `import cv2`, and
`from googletrans import Translator`. None of those are installed in the test
environment — the shims in `conftest.py` handle them.
"""

from __future__ import annotations

import pytest

from modules.transcript_srt import format_timestamp_srt


class TestBasicFormatting:
    def test_zero_seconds(self):
        assert format_timestamp_srt(0.0) == "00:00:00,000"

    def test_under_one_second_milliseconds_only(self):
        assert format_timestamp_srt(0.5) == "00:00:00,500"

    def test_exact_one_second(self):
        assert format_timestamp_srt(1.0) == "00:00:01,000"

    def test_minutes_and_seconds(self):
        # 75.0s -> 1 minute, 15 seconds.
        assert format_timestamp_srt(75.0) == "00:01:15,000"

    def test_hours_minutes_seconds(self):
        # 3661.0s -> 1h, 1m, 1s.
        assert format_timestamp_srt(3661.0) == "01:01:01,000"


class TestMillisecondPrecision:
    def test_fractional_seconds_become_milliseconds(self):
        # 1.234s -> 1 second, 234 ms.
        assert format_timestamp_srt(1.234) == "00:00:01,234"

    def test_ms_does_not_overflow_to_one_thousand(self):
        # Floating-point can produce 999.999... -> 999 (not 1000). Pin that
        # the millisecond field never reads "1000" in any output.
        result = format_timestamp_srt(1.9999999)
        # Whatever the implementation rounds to, ms part must be <= 999.
        ms_part = int(result.split(",")[1])
        assert 0 <= ms_part <= 999

    def test_ms_is_three_digits_with_leading_zeros(self):
        # 1.05s should produce "...,050", not "...,50".
        assert format_timestamp_srt(1.05) == "00:00:01,050"


class TestNegativeAndOutOfRange:
    def test_negative_input_clamps_to_zero(self):
        # The function's docstring + impl explicitly clamp via max(0, ...).
        assert format_timestamp_srt(-10.0) == "00:00:00,000"

    def test_very_small_negative(self):
        assert format_timestamp_srt(-0.001) == "00:00:00,000"


class TestTypeHandling:
    def test_int_input_works(self):
        # `float(seconds)` cast inside the function handles ints.
        assert format_timestamp_srt(60) == "00:01:00,000"


class TestFormatInvariants:
    def test_always_uses_comma_separator_for_milliseconds(self):
        # SRT spec requires comma. VTT uses dot. The pipeline writes SRT.
        result = format_timestamp_srt(123.456)
        assert "," in result
        assert "." not in result

    def test_always_returns_two_digit_hours_minutes_seconds(self):
        # Even at 00:00:00, leading zeros are present.
        result = format_timestamp_srt(0.0)
        h, m, rest = result.split(":")
        s, ms = rest.split(",")
        assert len(h) == 2
        assert len(m) == 2
        assert len(s) == 2
        assert len(ms) == 3

    @pytest.mark.parametrize(
        "seconds,expected",
        [
            (0.0, "00:00:00,000"),
            (0.001, "00:00:00,001"),
            # 59.999 happens to land on a binary fraction that survives
            # `(s - int(s)) * 1000` cleanly -> 999. Kept as a documented
            # boundary case.
            (59.999, "00:00:59,999"),
            (3600.0, "01:00:00,000"),
            # 3599.999 does NOT survive cleanly: IEEE 754 makes
            # (3599.999 - 3599) * 1000 = 998.999... -> int() = 998.
            # This is the current production behaviour; pin it. If you ever
            # change `format_timestamp_srt` to round (instead of truncate)
            # the millisecond component, update this expectation and bump a
            # changelog note: subtitles will shift by 1 ms on some inputs.
            (3599.999, "00:59:59,998"),
        ],
    )
    def test_parametrized_known_values(self, seconds, expected):
        assert format_timestamp_srt(seconds) == expected
