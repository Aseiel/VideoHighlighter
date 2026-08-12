"""
Tests for modules.edl — the timestamped cut list.

The value of an EDL is that it round-trips: the machine writes one, a person
edits it, and the render honours the edit. So the tests that matter most are
the ones about *not* losing information — a save/load cycle that keeps the
timestamps, and a parser that refuses a misspelled key instead of quietly
dropping the field it was meant to set.

Rendering is exercised against real ffmpeg with tiny clips; everything else is
pure text and needs neither.
"""

from __future__ import annotations

import subprocess

import pytest

from modules.app_paths import ffmpeg_exe
from modules.edl import (
    Cut,
    Edl,
    EdlError,
    edl_from_clips,
    format_time,
    load_edl,
    parse_edl,
    parse_time,
    quantise_to_music,
    render_edl,
    save_edl,
    validate_edl,
)

FFMPEG = ffmpeg_exe()


def _ffmpeg_ok() -> bool:
    try:
        return subprocess.run([FFMPEG, "-version"],
                              capture_output=True).returncode == 0
    except OSError:
        return False


def _clip(path, colour="red", duration=4.0):
    subprocess.run(
        [FFMPEG, "-y", "-v", "error",
         "-f", "lavfi", "-i", f"color=c={colour}:size=160x120:rate=30:duration={duration}",
         "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
         "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
         str(path)],
        check=True, capture_output=True)
    return str(path)


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("8", 8.0),
    ("8.5", 8.5),
    ("0:08", 8.0),
    ("1:23.5", 83.5),
    ("10:00", 600.0),
    ("1:02:03", 3723.0),
    ("1:02:03.5", 3723.5),
    (12, 12.0),
    (12.25, 12.25),
])
def test_timestamps_are_read_the_way_people_write_them(text, expected):
    assert parse_time(text) == pytest.approx(expected)


@pytest.mark.parametrize("bad", [
    "abc", "1:2:3:4", "", "-5", -5, "1:", ":30", "nan", float("nan"),
    float("inf"), True,
])
def test_nonsense_timestamps_are_refused(bad):
    with pytest.raises(EdlError):
        parse_time(bad)


@pytest.mark.parametrize("seconds,expected", [
    (0.0, "0:00"),
    (8.0, "0:08"),
    (83.5, "1:23.5"),
    (600.0, "10:00"),
    (3723.5, "1:02:03.5"),
    (3.63578, "0:03.636"),
    (4.2, "0:04.2"),
])
def test_timestamps_are_written_readably(seconds, expected):
    """Trailing zeros trimmed, so a round number stays clean and an awkward
    one keeps the digits it needs."""
    assert format_time(seconds) == expected


def test_timestamps_survive_a_round_trip():
    for seconds in (0.0, 8.0, 83.5, 3723.5, 3.63578, 4.23578):
        assert parse_time(format_time(seconds)) == pytest.approx(seconds, abs=0.001)


def test_a_bar_length_survives_the_round_trip_to_the_millisecond():
    """The regression that put a real render 1.7 s off the beat.

    A bar at 66 BPM is 3.63578 s. Written to one decimal it comes back as 3.6,
    and that 36 ms compounds across twenty cuts into half a bar of drift.
    """
    bar = (60.0 / 66.01) * 4

    assert parse_time(format_time(bar)) == pytest.approx(bar, abs=0.001)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

SIMPLE = """
title: Test
cuts:
  - source: a.mp4
    in: 0:02.5
    out: 0:08
    transition: crossfade
    transition_duration: 0.75
  - source: b.mp4
    in: 1:00
    out: 1:06
"""


def test_a_cut_list_parses_into_cuts():
    edl = parse_edl(SIMPLE)

    assert edl.title == "Test"
    assert len(edl.cuts) == 2
    assert edl.cuts[0].start == pytest.approx(2.5)
    assert edl.cuts[0].end == pytest.approx(8.0)
    assert edl.cuts[0].transition == "crossfade"
    assert edl.cuts[0].transition_duration == pytest.approx(0.75)
    assert edl.cuts[1].start == pytest.approx(60.0)
    assert edl.cuts[1].transition == "cut"


def test_reel_duration_accounts_for_the_overlap():
    """5.5 + 6.0 of footage joined by a 0.75s crossfade is 10.75s of reel, not
    11.5 — reporting the sum is how a 90 second target delivers 84."""
    edl = parse_edl(SIMPLE)

    assert edl.source_duration == pytest.approx(11.5)
    assert edl.duration == pytest.approx(10.75)


def test_a_final_transition_is_not_counted():
    """There is nothing after the last cut to blend into."""
    edl = parse_edl("""
cuts:
  - source: a.mp4
    out: 5
    transition: crossfade
    transition_duration: 1.0
""")
    assert edl.duration == pytest.approx(5.0)


@pytest.mark.parametrize("text,message", [
    ("cuts: []", "at least one cut"),
    ("title: x", "needs a 'cuts:' list"),
    ("", "empty"),
    ("cuts:\n  - in: 0\n    out: 5\n", "no source"),
    ("cuts:\n  - source: a.mp4\n", "no 'out' time"),
    ("cuts:\n  - source: a.mp4\n    in: 8\n    out: 4\n", "must come after"),
    ("cuts:\n  - source: a.mp4\n    out: 5\n    transition: wibble\n", "unknown transition"),
    ("cuts:\n  - source: a.mp4\n    outt: 5\n", "did you mean 'out'"),
    ("titel: x\ncuts:\n  - source: a.mp4\n    out: 5\n", "did you mean 'title'"),
    ("cuts: 5", "must be a list"),
    ("cuts:\n  - just a string\n", "must be a mapping"),
    ("- a\n- b\n", "must be a mapping"),
    ("version: 99\ncuts:\n  - source: a.mp4\n    out: 5\n", "not supported"),
    ("cuts:\n  - source: a.mp4\n    out: 5\n    transition_duration: soon\n", "must be a number"),
    ("cuts:\n  - source: a.mp4\n    out: 5\n    transition_duration: -1\n", "non-negative"),
    ("cuts: [\n", "not valid YAML"),
])
def test_a_broken_cut_list_says_what_is_wrong(text, message):
    with pytest.raises(EdlError, match=message):
        parse_edl(text)


def test_an_error_names_the_cut_it_is_about():
    """A cut list is a column of near-identical entries; "invalid duration" on
    its own sends you reading all of them."""
    with pytest.raises(EdlError, match="cut 2"):
        parse_edl("""
cuts:
  - source: a.mp4
    out: 5
  - source: b.mp4
    in: 9
    out: 3
""")


# ---------------------------------------------------------------------------
# Round trip — the reason the format exists
# ---------------------------------------------------------------------------

def test_save_and_load_keeps_every_field(tmp_path):
    edl = Edl(
        title="Round trip",
        cuts=[Cut("a.mp4", 2.5, 8.0, "crossfade", 0.75, label="Opening"),
              Cut("b.mp4", 60.0, 66.0, "dip_to_black", 1.0),
              Cut("c.mp4", 0.0, 4.0)],
        music="track.mp3", music_mode="duck", music_volume=0.6,
        width=1920, height=1080, fps=30, crf=18)
    path = str(tmp_path / "film.edl.yaml")

    save_edl(edl, path)
    loaded = load_edl(path)

    assert loaded.title == edl.title
    assert loaded.music == "track.mp3"
    assert loaded.music_mode == "duck"
    assert loaded.music_volume == pytest.approx(0.6)
    assert (loaded.width, loaded.height, loaded.fps, loaded.crf) == (1920, 1080, 30, 18)
    assert len(loaded.cuts) == 3
    for before, after in zip(edl.cuts, loaded.cuts):
        assert after.source == before.source
        assert after.start == pytest.approx(before.start, abs=0.05)
        assert after.end == pytest.approx(before.end, abs=0.05)
    assert loaded.cuts[0].transition == "crossfade"
    assert loaded.cuts[1].transition == "dip_to_black"
    assert loaded.cuts[0].label == "Opening"


def test_a_saved_cut_list_is_readable(tmp_path):
    """It is meant to be opened and edited, so the timestamps must be clock
    times rather than floats."""
    edl = Edl(title="Readable",
              cuts=[Cut("a.mp4", 83.5, 90.0), Cut("b.mp4", 0.0, 4.0)])
    path = str(tmp_path / "film.edl.yaml")

    save_edl(edl, path)
    text = open(path, encoding="utf-8").read()

    assert "in: 1:23.5" in text
    assert "out: 1:30" in text
    assert text.lstrip().startswith("#"), "no header explaining the file"


def test_the_last_cut_gets_no_transition_line(tmp_path):
    """There is nothing to blend into, and an ignored field in a file people
    edit is an invitation to set it and wonder why nothing happened."""
    edl = Edl(cuts=[Cut("a.mp4", 0, 4, "crossfade", 0.5),
                    Cut("b.mp4", 0, 4, "crossfade", 0.5)])
    path = str(tmp_path / "film.edl.yaml")

    save_edl(edl, path)
    body = open(path, encoding="utf-8").read().split("cuts:")[1]

    assert body.count("transition:") == 1


def test_a_non_utf8_cut_list_is_an_edl_error(tmp_path):
    path = tmp_path / "film.edl.yaml"
    path.write_bytes("title: Film\ncuts:\n  - source: a.mp4\n    out: 5\n".encode("utf-16"))

    with pytest.raises(EdlError, match="not UTF-8"):
        load_edl(str(path))


# ---------------------------------------------------------------------------
# Building one from a finished run
# ---------------------------------------------------------------------------

def test_clips_become_whole_file_cuts(tmp_path):
    paths = [str(tmp_path / "a.mp4"), str(tmp_path / "b.mp4")]
    for p in paths:
        open(p, "wb").write(b"x")

    edl = edl_from_clips(paths, title="From a run", transition="crossfade",
                         probe=False)

    assert edl.title == "From a run"
    assert [c.source for c in edl.cuts] == paths
    assert all(c.start == 0.0 for c in edl.cuts)
    assert edl.cuts[0].transition == "crossfade"
    assert edl.cuts[-1].transition == "cut", "nothing follows the last cut"
    assert edl.cuts[0].label == "a"


def test_building_from_no_clips_is_an_empty_list():
    assert edl_from_clips([]).cuts == []


# ---------------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------------

def test_warnings_flag_missing_sources_without_refusing_them(tmp_path):
    """The file may be about to be plugged in; refusing to even describe the
    edit would be obnoxious."""
    edl = Edl(cuts=[Cut(str(tmp_path / "gone.mp4"), 0, 5)],
              music=str(tmp_path / "nomusic.mp3"))

    warnings = validate_edl(edl)

    assert any("gone.mp4" in w for w in warnings)
    assert any("music file" in w for w in warnings)


@pytest.mark.skipif(not _ffmpeg_ok(), reason="ffmpeg not available")
def test_warnings_flag_a_cut_past_the_end_of_its_source(tmp_path):
    source = _clip(tmp_path / "a.mp4", duration=4.0)

    warnings = validate_edl(Edl(cuts=[Cut(source, 0.0, 30.0)]))

    assert any("past the end" in w for w in warnings)


@pytest.mark.skipif(not _ffmpeg_ok(), reason="ffmpeg not available")
def test_a_valid_cut_list_warns_about_nothing(tmp_path):
    source = _clip(tmp_path / "a.mp4", duration=4.0)

    assert validate_edl(Edl(cuts=[Cut(source, 0.5, 3.5)])) == []


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _ffmpeg_ok(), reason="ffmpeg not available")
def test_rendering_honours_the_timestamps(tmp_path):
    """Two 2-second pieces taken out of 6-second sources make a 4-second reel —
    which is the whole claim of the format."""
    from modules.video_probe import probe_video

    a = _clip(tmp_path / "a.mp4", "red", duration=6.0)
    b = _clip(tmp_path / "b.mp4", "blue", duration=6.0)
    edl = Edl(cuts=[Cut(a, 1.0, 3.0), Cut(b, 2.0, 4.0)], width=64, height=48)
    out = str(tmp_path / "film.mp4")

    render_edl(edl, out, mode="cpu", log_fn=lambda *_: None)

    assert probe_video(out)["duration"] == pytest.approx(4.0, abs=0.4)


@pytest.mark.skipif(not _ffmpeg_ok(), reason="ffmpeg not available")
def test_rendering_applies_the_transition(tmp_path):
    from modules.video_probe import probe_video

    a = _clip(tmp_path / "a.mp4", "red", duration=6.0)
    b = _clip(tmp_path / "b.mp4", "blue", duration=6.0)
    edl = Edl(cuts=[Cut(a, 0.0, 3.0, "crossfade", 1.0), Cut(b, 0.0, 3.0)],
              width=64, height=48)
    out = str(tmp_path / "film.mp4")

    render_edl(edl, out, mode="cpu", log_fn=lambda *_: None)

    assert probe_video(out)["duration"] == pytest.approx(5.0, abs=0.4)


def test_rendering_a_missing_source_fails_before_any_work(tmp_path):
    edl = Edl(cuts=[Cut(str(tmp_path / "gone.mp4"), 0, 5)])

    with pytest.raises(EdlError, match="sources are missing"):
        render_edl(edl, str(tmp_path / "out.mp4"), log_fn=lambda *_: None)


def test_rendering_an_empty_cut_list_is_refused(tmp_path):
    with pytest.raises(EdlError, match="cut list is empty"):
        render_edl(Edl(), str(tmp_path / "out.mp4"), log_fn=lambda *_: None)


# ---------------------------------------------------------------------------
# Quantising to the music
# ---------------------------------------------------------------------------

class _Analysis:
    """Just the two fields quantise_to_music reads."""

    def __init__(self, bpm, meter=4):
        self.beat_interval = 60.0 / bpm if bpm else 0.0
        self.meter = meter


def test_cuts_are_rounded_to_whole_bars(tmp_path):
    """At 120 BPM a 4/4 bar is 2 s, so 5.4 s becomes 6 s and 3.1 s becomes 4.

    Durations rather than positions: a reel plays back to back, so making each
    clip a whole number of bars is what puts every cut on a downbeat.
    """
    edl = Edl(cuts=[Cut("a.mp4", 0.0, 5.4), Cut("b.mp4", 0.0, 3.1)])

    aligned = quantise_to_music(edl, _Analysis(120), log_fn=lambda *_: None)

    assert aligned.cuts[0].duration == pytest.approx(6.0)
    assert aligned.cuts[1].duration == pytest.approx(4.0)


def test_quantising_to_the_beat_instead_of_the_bar():
    edl = Edl(cuts=[Cut("a.mp4", 0.0, 5.4)])

    aligned = quantise_to_music(edl, _Analysis(120), unit="beat",
                                log_fn=lambda *_: None)

    assert aligned.cuts[0].duration == pytest.approx(5.5)


def test_every_cut_lands_on_a_downbeat_once_quantised():
    """The property that matters: cumulative time is always a whole number of
    bars, so cut 2 and cut 3 are on the beat as much as cut 1 is."""
    bar = 2.0
    edl = Edl(cuts=[Cut("a.mp4", 0.0, 5.4), Cut("b.mp4", 0.0, 3.1),
                    Cut("c.mp4", 0.0, 7.7)])

    aligned = quantise_to_music(edl, _Analysis(120), log_fn=lambda *_: None)

    running = 0.0
    for cut in aligned.cuts:
        running += cut.duration
        assert running % bar == pytest.approx(0.0, abs=0.01)


def test_transitions_do_not_knock_the_cuts_off_the_beat():
    """The bug the first real render shipped with.

    A transition overlaps two clips, so the reel advances by
    ``duration - transition`` rather than by ``duration``. Quantising the
    duration alone therefore leaves every join progressively further off: at a
    3.64 s bar with 0.6 s crossfades the third join measured 1.8 s out, which
    is half a bar. Each clip has to carry its own transition on top so the
    *advance* is the bar.
    """
    bar = 2.0
    blend = 0.4
    edl = Edl(cuts=[Cut("a.mp4", 0.0, 5.4, "crossfade", blend),
                    Cut("b.mp4", 0.0, 3.1, "crossfade", blend),
                    Cut("c.mp4", 0.0, 7.7, "crossfade", blend),
                    Cut("d.mp4", 0.0, 4.4)])

    aligned = quantise_to_music(edl, _Analysis(120), log_fn=lambda *_: None)

    # Where each transition begins in the finished reel.
    start = 0.0
    for i, cut in enumerate(aligned.cuts[:-1]):
        start += cut.duration - cut.transition_duration
        assert start % bar == pytest.approx(0.0, abs=0.01), (
            f"join {i + 1} starts at {start:.3f}s, off the {bar}s bar")


def test_alignment_survives_being_written_to_disk_and_read_back(tmp_path):
    """The end-to-end version of the two bugs above, and the one that would
    have caught both.

    Quantising in memory is not the claim; the claim is that the *rendered*
    reel lands on the beat, and everything between here and the renderer goes
    through the file. A rounding-friendly bar (120 BPM) would pass while
    hiding the precision loss, so this uses 66.01 BPM — the track that exposed
    it — where a bar is 3.63578 s.
    """
    edl = Edl(cuts=[Cut(f"{n}.mp4", 0.0, 6.03, "crossfade", 0.6)
                    for n in "abcdefgh"])
    edl.cuts[-1].transition = "cut"

    aligned = quantise_to_music(edl, _Analysis(66.01), log_fn=lambda *_: None)
    path = str(tmp_path / "film.edl.yaml")
    save_edl(aligned, path)
    reloaded = load_edl(path)

    bar = (60.0 / 66.01) * 4
    start = 0.0
    for i, cut in enumerate(reloaded.cuts[:-1]):
        start += cut.duration - (cut.transition_duration
                                 if cut.transition != "cut" else 0.0)
        off = abs(start - round(start / bar) * bar)
        assert off < 0.01, (
            f"join {i + 1} is {off * 1000:.0f}ms off the bar after a round trip")


@pytest.mark.skipif(not _ffmpeg_ok(), reason="ffmpeg not available")
def test_a_clip_with_no_room_for_the_blend_cuts_hard_and_keeps_the_grid(tmp_path):
    """The last thing keeping a real render off the beat.

    A 4 s clip against a 3.64 s bar has room for the bar but not for the bar
    plus a 0.6 s blend. Keeping its raw 4 s puts every later cut off the grid;
    dropping to zero bars loses the clip. Cutting hard at exactly one bar
    keeps both the clip and the alignment.
    """
    short = _clip(tmp_path / "short.mp4", duration=4.0)
    long = _clip(tmp_path / "long.mp4", duration=8.0)
    edl = Edl(cuts=[Cut(short, 0.0, 4.0, "crossfade", 0.6),
                    Cut(long, 0.0, 6.03, "crossfade", 0.6),
                    Cut(long, 0.0, 6.03)])

    aligned = quantise_to_music(edl, _Analysis(66.01), log_fn=lambda *_: None)

    bar = (60.0 / 66.01) * 4
    assert aligned.cuts[0].transition == "cut"
    assert aligned.cuts[0].duration == pytest.approx(bar, abs=0.01)
    start = 0.0
    for i, cut in enumerate(aligned.cuts[:-1]):
        start += cut.duration - (cut.transition_duration
                                 if cut.transition != "cut" else 0.0)
        off = abs(start - round(start / bar) * bar)
        assert off < 0.01, f"join {i + 1} is {off * 1000:.0f}ms off the bar"


def test_a_hard_cut_gets_no_transition_allowance():
    """There is no overlap to compensate for, so the clip is exactly bars."""
    edl = Edl(cuts=[Cut("a.mp4", 0.0, 5.4, "cut", 0.5),
                    Cut("b.mp4", 0.0, 3.1)])

    aligned = quantise_to_music(edl, _Analysis(120), log_fn=lambda *_: None)

    assert aligned.cuts[0].duration == pytest.approx(6.0)


def test_the_last_clip_gets_no_transition_allowance():
    """Nothing follows it to overlap with.

    5.8 s is used rather than 5.4 so neither clip lands on a .5 rounding tie,
    where Python rounds to even and the arithmetic stops being obvious.
    """
    edl = Edl(cuts=[Cut("a.mp4", 0.0, 5.8, "crossfade", 0.4),
                    Cut("b.mp4", 0.0, 5.8, "crossfade", 0.4)])

    aligned = quantise_to_music(edl, _Analysis(120), log_fn=lambda *_: None)

    assert aligned.cuts[0].duration == pytest.approx(6.4)   # 3 bars + blend
    assert aligned.cuts[1].duration == pytest.approx(6.0)   # 3 bars


@pytest.mark.skipif(not _ffmpeg_ok(), reason="ffmpeg not available")
def test_a_cut_that_cannot_grow_drops_to_the_next_whole_bar(tmp_path):
    """The case that made the first real render come out off the beat.

    A 6.03 s clip against a 3.64 s bar rounds to two bars, has nowhere near
    that much footage, and must become *one* bar. Clamping it to 6.03 s instead
    keeps two extra seconds and puts every following cut off the grid, which is
    the one thing quantising exists to prevent.
    """
    source = _clip(tmp_path / "a.mp4", duration=6.03)
    edl = Edl(cuts=[Cut(source, 0.0, 6.03)])

    aligned = quantise_to_music(edl, _Analysis(66.01), log_fn=lambda *_: None)

    bar = (60.0 / 66.01) * 4
    assert aligned.cuts[0].duration == pytest.approx(bar, abs=0.01)


@pytest.mark.skipif(not _ffmpeg_ok(), reason="ffmpeg not available")
def test_a_cut_is_never_lengthened_past_its_source(tmp_path):
    """Rounding up into footage that does not exist would render black."""
    source = _clip(tmp_path / "a.mp4", duration=3.7)
    edl = Edl(cuts=[Cut(source, 0.0, 3.6)])

    aligned = quantise_to_music(edl, _Analysis(120), log_fn=lambda *_: None)

    assert aligned.cuts[0].end <= 3.75


@pytest.mark.skipif(not _ffmpeg_ok(), reason="ffmpeg not available")
def test_a_clip_shorter_than_one_bar_keeps_its_own_length(tmp_path):
    """Below a whole unit there is nothing to align to, and a clip trimmed to
    nothing is worse than one that breaks the pattern."""
    source = _clip(tmp_path / "a.mp4", duration=1.2)
    edl = Edl(cuts=[Cut(source, 0.0, 1.2)])

    aligned = quantise_to_music(edl, _Analysis(120), log_fn=lambda *_: None)

    assert aligned.cuts[0].duration == pytest.approx(1.2, abs=0.05)


def test_min_units_stops_a_short_clip_rounding_to_nothing():
    edl = Edl(cuts=[Cut("a.mp4", 0.0, 0.4)])

    aligned = quantise_to_music(edl, _Analysis(120), log_fn=lambda *_: None)

    assert aligned.cuts[0].duration == pytest.approx(2.0)


def test_transition_lengths_can_come_from_the_bar():
    edl = Edl(cuts=[Cut("a.mp4", 0.0, 4.0, "crossfade", 0.5),
                    Cut("b.mp4", 0.0, 4.0)])

    aligned = quantise_to_music(edl, _Analysis(120), transition_bars=0.25,
                                log_fn=lambda *_: None)

    assert aligned.cuts[0].transition_duration == pytest.approx(0.5)


def test_no_tempo_leaves_the_cuts_alone():
    """A track that could not be analysed must not silently retime the edit."""
    edl = Edl(cuts=[Cut("a.mp4", 0.0, 5.4)])

    aligned = quantise_to_music(edl, _Analysis(0), log_fn=lambda *_: None)

    assert aligned.cuts[0].duration == pytest.approx(5.4)


def test_quantising_does_not_modify_the_input():
    edl = Edl(cuts=[Cut("a.mp4", 0.0, 5.4)])

    quantise_to_music(edl, _Analysis(120), log_fn=lambda *_: None)

    assert edl.cuts[0].end == pytest.approx(5.4)


def test_an_unknown_unit_is_refused():
    with pytest.raises(EdlError, match="unknown quantise unit"):
        quantise_to_music(Edl(cuts=[Cut("a.mp4", 0, 4)]), _Analysis(120),
                          unit="phrase", log_fn=lambda *_: None)
