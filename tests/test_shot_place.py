"""
Tests for modules.shot_place — which clips were shot from the same spot.

The grouping rules are pinned against the numbers that motivated them, taken
from a real shoot: four pairs of clips filmed 11-127 m apart within two minutes
of each other, and one pair 479 m apart that must stay separate because they
are two genuinely different views of the same valley. Those two facts are what
the thresholds sit between, so they are what the tests assert.

Reading a file's own metadata needs ffprobe and is exercised end to end
elsewhere; what is pinned here is everything that decides the edit.
"""

from __future__ import annotations

import datetime as dt

import pytest

from modules.shot_place import (
    PLACE_METRES,
    Place,
    Track,
    distance,
    group,
    read_track,
)


def _at(minute: float, lat: float = None, lon: float = None,
        name: str = "") -> Place:
    """A clip shot ``minute`` minutes into the day, optionally located."""
    return Place(
        path=name or f"clip{minute}.mp4",
        when=dt.datetime(2026, 8, 8, 8, 0, tzinfo=dt.timezone.utc)
        + dt.timedelta(minutes=minute),
        latitude=lat, longitude=lon,
        source="tag" if lat is not None else "time")


def _group(places, **kw):
    return group({p.path: p for p in places}, log_fn=lambda *_: None, **kw)


# ---------------------------------------------------------------------------
# Distance
# ---------------------------------------------------------------------------

def test_distance_matches_the_real_pairs():
    """The four pairs this module was built for, and the one that must not
    be merged with them."""
    # GX013767 and GX013768, eleven seconds apart on the same ridge.
    assert distance(_at(0, 53.5155, -1.9696),
                    _at(0, 53.5154, -1.9696)) == pytest.approx(11, abs=8)
    # GX013762 and GX013764: the same valley from two different vantage
    # points, which are worth having both of.
    assert distance(_at(0, 53.5218, -1.9950),
                    _at(0, 53.5202, -1.9883)) == pytest.approx(479, abs=30)


def test_distance_is_infinite_without_a_position():
    assert distance(_at(0), _at(1, 53.5, -1.9)) == float("inf")
    assert distance(_at(0), _at(1)) == float("inf")


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------

def test_clips_from_one_spot_are_one_place():
    a, b = _at(0, 53.5155, -1.9696, "a"), _at(0.2, 53.5154, -1.9696, "b")

    numbers = _group([a, b])

    assert numbers["a"] == numbers["b"]


def test_two_vantage_points_on_one_valley_stay_separate():
    """479 m apart. Position says different places and position is right —
    what makes these look alike is the subject, which modules.shot_look
    handles and this module deliberately does not guess at."""
    a, b = _at(0, 53.5218, -1.9950, "a"), _at(2.7, 53.5202, -1.9883, "b")

    numbers = _group([a, b])

    assert numbers["a"] != numbers["b"]


def test_time_decides_when_position_is_unknown():
    """The case that matters most and needs no GPS at all: two clips a few
    seconds apart are the same setup whatever else is true."""
    a, b, c = _at(0, name="a"), _at(0.3, name="b"), _at(40, name="c")

    numbers = _group([a, b, c])

    assert numbers["a"] == numbers["b"]
    assert numbers["c"] != numbers["a"]


def test_position_beats_time_when_both_are_known():
    """Standing in one spot for ten minutes is one place, however long the
    gap; running for two minutes is not, however short it is."""
    still = _group([_at(0, 53.5155, -1.9696, "a"),
                    _at(9, 53.5155, -1.9697, "b")])
    assert still["a"] == still["b"], "a long stop in one spot is one place"

    moving = _group([_at(0, 53.5000, -1.9000, "a"),
                     _at(1.5, 53.5200, -1.9500, "b")])
    assert moving["a"] != moving["b"], "two km apart is not one place"


def test_a_long_route_does_not_chain_into_one_place():
    """Stopping every hundred metres along a ridge must not link into a
    single place stretching for miles, which is what transitive grouping
    would do — each clip is compared to the place's first member, not its
    last."""
    places = [_at(i, 53.5 + i * 0.001, -1.9, f"c{i}") for i in range(8)]

    numbers = _group(places)

    assert len(set(numbers.values())) > 1


def test_every_clip_gets_a_place_even_with_nothing_to_go_on():
    """A shoot whose files carry neither a time nor a position still cuts."""
    blanks = [Place(path=f"c{i}.mp4") for i in range(4)]

    numbers = _group(blanks)

    assert set(numbers) == {p.path for p in blanks}
    assert all(isinstance(n, int) for n in numbers.values())


def test_grouping_nothing_is_not_an_error():
    assert _group([]) == {}


# ---------------------------------------------------------------------------
# Tracks
# ---------------------------------------------------------------------------

GPX = """<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" xmlns="http://www.topografix.com/GPX/1/1">
 <trk><trkseg>
  <trkpt lat="53.5000" lon="-1.9000"><ele>300</ele><time>2026-08-08T08:00:00.000Z</time></trkpt>
  <trkpt lat="53.5100" lon="-1.9100"><ele>310</ele><time>2026-08-08T08:10:00.000Z</time></trkpt>
  <trkpt lat="53.5200" lon="-1.9200"><ele>320</ele><time>2026-08-08T08:20:00.000Z</time></trkpt>
 </trkseg></trk>
</gpx>
"""


def _track(tmp_path, text=GPX):
    path = tmp_path / "run.gpx"
    path.write_text(text, encoding="utf-8")
    return read_track(str(path), log_fn=lambda *_: None)


def test_a_track_is_read_with_its_times(tmp_path):
    track = _track(tmp_path)

    assert len(track.points) == 3
    assert track.points[0][1] == pytest.approx(53.5)


def test_a_track_answers_where_you_were(tmp_path):
    track = _track(tmp_path)

    found = track.at(dt.datetime(2026, 8, 8, 8, 10, tzinfo=dt.timezone.utc))

    assert found == pytest.approx((53.51, -1.91))


def test_a_track_refuses_a_time_it_does_not_cover(tmp_path):
    """Silence is the only honest answer for a clip filmed before the watch
    was started — inventing the nearest point would place it a valley away."""
    track = _track(tmp_path)

    assert track.at(dt.datetime(2026, 8, 8, 12, 0, tzinfo=dt.timezone.utc)) is None


def test_a_naive_timestamp_is_read_as_utc(tmp_path):
    """What ffmpeg writes into creation_time, and what GPX requires."""
    track = _track(tmp_path)

    assert track.at(dt.datetime(2026, 8, 8, 8, 10)) is not None


def test_an_unreadable_track_costs_the_track_and_nothing_else(tmp_path):
    bad = tmp_path / "broken.gpx"
    bad.write_text("this is not xml <<<", encoding="utf-8")

    assert not read_track(str(bad), log_fn=lambda *_: None)
    assert not read_track(str(tmp_path / "absent.gpx"), log_fn=lambda *_: None)
    assert not Track().at(dt.datetime.now(dt.timezone.utc))


def test_the_place_radius_sits_between_the_two_real_clusters():
    """A regression guard on the constant itself: the pairs that are one
    setup are 11-127 m apart and the nearest pair that is not is 479 m, so
    the threshold has room on both sides rather than balancing on an edge."""
    assert 130 < PLACE_METRES < 450
