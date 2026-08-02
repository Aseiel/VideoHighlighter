"""Tests for chapters reaching the report — the wiring, not the arithmetic.

`tests/test_chapter_compare.py` pins down what the numbers mean. This file pins
down that they survive the trip into `build_report` and out through both
renderers, and — the part most likely to break silently — that a report built
*without* chapters is byte-for-byte the report that existed before they did.
"""
from __future__ import annotations

import json

import numpy as np

from modules.highlight_report import build_report, render_html, render_text


def _signals(n=600, **overrides):
    keys = ("scene", "motion_event", "motion_peak", "audio",
            "keyword", "object", "action")
    sig = {k: np.zeros(n) for k in keys}
    for key, points in overrides.items():
        for sec, value in points.items():
            sig[key][sec] = value
    return sig


def _chapters():
    return [
        {"number": 1, "start": 0.0, "end": 200.0, "duration": 200.0,
         "timestamp": "0:00:00", "title": "Chapter 1", "shots": 60,
         "pace": "fast-cut", "boundary_score": 0.0, "method": "visual"},
        {"number": 2, "start": 200.0, "end": 600.0, "duration": 400.0,
         "timestamp": "0:03:20", "title": "Chapter 2", "shots": 20,
         "pace": "held", "boundary_score": 0.62, "method": "visual"},
    ]


def _report(**kw):
    signals = _signals(audio={10: 5.0, 30: 4.0, 300: 3.0})
    score = sum(signals.values())
    return build_report(
        video_path="a.mp4", video_duration=600.0, score=score, signals=signals,
        segments=[(8.0, 18.0), (28.0, 38.0), (298.0, 308.0)], **kw)


class TestReportCarriesChapters:
    def test_chapters_land_in_the_record(self):
        rep = _report(chapters=_chapters())
        assert len(rep["chapters"]) == 2
        assert rep["chapters"][0]["clips"] == 2
        assert rep["chapters"][1]["clips"] == 1

    def test_each_clip_is_tagged_with_its_chapter(self):
        rep = _report(chapters=_chapters())
        assert [e["chapter"] for e in rep["segments"]] == [1, 1, 2]

    def test_cut_share_is_measured_against_runtime(self):
        rep = _report(chapters=_chapters())
        # Chapter 1 is a third of the runtime and supplied two of three clips.
        assert rep["chapters"][0]["cut_share_lift"] > 1.5
        assert rep["chapters"][1]["cut_share_lift"] < 1.0

    def test_the_record_stays_serialisable(self):
        rep = _report(chapters=_chapters())
        json.dumps(rep)


class TestAbsentChaptersChangeNothing:
    def test_no_chapters_is_an_empty_list_not_a_crash(self):
        rep = _report()
        assert rep["chapters"] == []

    def test_no_clip_is_tagged(self):
        rep = _report()
        assert all("chapter" not in e for e in rep["segments"])

    def test_neither_renderer_mentions_chapters(self):
        rep = _report()
        assert "chapter" not in render_text(rep).lower()
        assert "in chapters" not in render_html(rep).lower()

    def test_a_malformed_chapter_list_costs_only_the_section(self):
        """A bad partition must not take the rest of the report down with it."""
        rep = _report(chapters=[{"number": 1}])       # missing start/end
        assert rep["chapters"] == []
        assert len(rep["segments"]) == 3


class TestRenderers:
    def test_text_lists_the_chapters_and_their_clips(self):
        out = render_text(_report(chapters=_chapters()))
        assert "--- The video in chapters ---" in out
        assert "Chapter 1" in out and "Chapter 2" in out
        assert "[1], [2]" in out          # both clips filed under chapter 1

    def test_text_says_when_a_chapter_contributed_nothing(self):
        chapters = _chapters()
        rep = build_report(
            video_path="a.mp4", video_duration=600.0,
            score=np.zeros(600), signals=_signals(),
            segments=[(8.0, 18.0)], chapters=chapters)
        assert "Nothing from this chapter was selected." in render_text(rep)

    def test_html_renders_the_section_and_the_strip(self):
        page = render_html(_report(chapters=_chapters()))
        assert "The video in chapters" in page
        assert 'class="chapstrip"' in page
        assert 'class="chap"' in page

    def test_html_escapes_a_caller_supplied_title(self):
        chapters = _chapters()
        chapters[0]["title"] = '<script>alert(1)</script>'
        page = render_html(_report(chapters=chapters))
        assert "<script>alert(1)</script>" not in page
        assert "&lt;script&gt;" in page

    def test_html_is_still_one_self_contained_page(self):
        page = render_html(_report(chapters=_chapters()))
        assert page.count("<!doctype html>") == 1
        assert "http://" not in page and "https://" not in page


class TestDescriptions:
    def test_a_concentrated_chapter_says_so(self):
        from modules.highlight_prose import describe_chapter

        rep = _report(chapters=_chapters())
        lines = describe_chapter(rep["chapters"][0])
        assert any("of the cut" in line for line in lines)

    def test_an_undistinguished_chapter_says_that_instead(self):
        from modules.highlight_prose import describe_chapter

        # Parity on every axis: same length, same shot rate, one clip each.
        chapters = [
            {"number": 1, "start": 0.0, "end": 300.0, "duration": 300.0,
             "timestamp": "0:00:00", "title": "Chapter 1", "shots": 30,
             "pace": "steady", "boundary_score": 0.0, "method": "visual"},
            {"number": 2, "start": 300.0, "end": 600.0, "duration": 300.0,
             "timestamp": "0:05:00", "title": "Chapter 2", "shots": 30,
             "pace": "steady", "boundary_score": 0.5, "method": "visual"},
        ]
        signals = _signals(audio={10: 5.0, 310: 5.0})
        rep = build_report(
            video_path="a.mp4", video_duration=600.0,
            score=sum(signals.values()), signals=signals,
            segments=[(8.0, 18.0), (308.0, 318.0)], chapters=chapters)

        lines = describe_chapter(rep["chapters"][0])
        assert lines == ["Nothing here separates it from the rest of the video."]

    def test_the_headline_reports_concentration(self):
        from modules.highlight_prose import summarise_chapter_run

        rep = _report(chapters=_chapters())
        sentence = summarise_chapter_run(rep["chapters"])
        assert "2 chapters" in sentence
        assert "real cut" not in sentence      # that claim belongs to the note

    def test_a_single_chapter_video_is_described_as_undivided(self):
        from modules.highlight_prose import summarise_chapter_run

        sentence = summarise_chapter_run([
            {"number": 1, "start": 0.0, "end": 120.0, "title": "Chapter 1",
             "method": "single", "clips": 1, "cut_share_pct": 100.0},
        ])
        assert "not divided" in sentence
