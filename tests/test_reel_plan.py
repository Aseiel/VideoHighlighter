"""
Tests for modules.reel_plan — the short-form story arrangement.

What is worth pinning here is not that the function returns cuts, but that the
three claims it makes are true of the result: that the reel lands near the
length that was asked for, that its cutting rate falls in the band the chosen
pace names, and that the structure is actually a structure — the hook first and
alone, the payoff held longer than the body, everything else in the order it
was shot.

Sources are generated once as tiny clips, since the planner only reads their
durations.
"""

from __future__ import annotations

import subprocess

import pytest

from modules.app_paths import ffmpeg_exe
from modules.reel_plan import (
    LENGTHS,
    MIN_SHOT,
    PACES,
    STRUCTURE,
    cuts_per_minute,
    describe_plan,
    minimum_duration,
    plan_reel,
)

FFMPEG = ffmpeg_exe()

# The bands the pace names claim, as cuts per minute.
SPEC_BANDS = {
    "calm": (10, 20),
    "vlog": (15, 30),
    "energetic": (25, 60),
    "intense": (40, 120),
}


def _ffmpeg_ok() -> bool:
    try:
        return subprocess.run([FFMPEG, "-version"],
                              capture_output=True).returncode == 0
    except OSError:
        return False


pytestmark = pytest.mark.skipif(not _ffmpeg_ok(), reason="ffmpeg not available")


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    """Twelve 6-second clips — roughly what a short shoot yields after the
    engine has picked its highlights."""
    root = tmp_path_factory.mktemp("reel")
    made = []
    for i in range(12):
        path = root / f"clip{i:02d}.mp4"
        subprocess.run(
            [FFMPEG, "-y", "-v", "error",
             "-f", "lavfi", "-i", f"color=c=gray:size=64x48:rate=30:duration=6",
             "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
             "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
             str(path)],
            check=True, capture_output=True)
        made.append(str(path))
    return made


class _Analysis:
    """Only the two fields the planner reads off a music analysis."""

    def __init__(self, bpm=120.0, meter=4):
        self.beat_interval = 60.0 / bpm if bpm else 0.0
        self.meter = meter


# ---------------------------------------------------------------------------
# The three claims
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("duration", [15, 20, 24, 30, 45, 60])
@pytest.mark.parametrize("pace", ["energetic", "intense"])
def test_the_reel_is_about_as_long_as_asked(clips, duration, pace):
    """Within about a tenth. Somebody choosing 24 seconds for a Reel has a
    reason for that number."""
    reel = plan_reel(clips, duration=duration, pace=pace, log_fn=lambda *_: None)

    assert reel.duration == pytest.approx(duration, rel=0.12)


@pytest.mark.parametrize("pace", list(PACES))
def test_the_cutting_rate_matches_the_pace_it_names(clips, pace):
    """A preset called "energetic" that cuts fifteen times a minute is lying
    about the only thing it exists to control."""
    reel = plan_reel(clips, duration=45, pace=pace, log_fn=lambda *_: None)
    low, high = SPEC_BANDS[pace]

    assert low - 3 <= cuts_per_minute(reel) <= high + 3


def test_the_structure_is_present_and_in_order(clips):
    reel = plan_reel(clips, duration=24, pace="energetic", log_fn=lambda *_: None)
    labels = [c.label for c in reel.cuts]

    assert labels[0] == "Hook"
    assert labels.count("Hook") == 1, "the hook is one shot, not a section"
    assert labels[-1] == "Payoff"
    # Sections appear in order and do not interleave.
    order = [name for i, name in enumerate(labels) if i == 0 or labels[i - 1] != name]
    assert order == ["Hook", "Context", "Escalation", "Payoff"]


def test_the_payoff_is_held_longer_than_the_body(clips):
    """An ending cut at the body's rhythm does not read as an ending."""
    reel = plan_reel(clips, duration=24, pace="energetic", log_fn=lambda *_: None)
    body = [c.duration for c in reel.cuts if c.label == "Escalation"]
    payoff = [c.duration for c in reel.cuts if c.label == "Payoff"]

    assert min(payoff) >= max(body) * 1.5


def test_the_hook_is_the_highest_scoring_clip_not_the_first(clips):
    """The opening shot has to be the most striking thing available. Shooting
    order says nothing about that."""
    scores = {clips[7]: 99.0, clips[0]: 1.0}

    reel = plan_reel(clips, duration=24, pace="energetic", scores=scores,
                     log_fn=lambda *_: None)

    assert reel.cuts[0].source == clips[7]


def test_everything_after_the_hook_keeps_shooting_order(clips):
    """Progression is what makes a sequence read as a story rather than a
    shuffle, and shooting order is the only progression the footage has.

    Twenty seconds needs ten shots from twelve clips, so nothing is reused and
    the order is simply forward.
    """
    reel = plan_reel(clips, duration=20, pace="energetic", log_fn=lambda *_: None)
    positions = [clips.index(c.source) for c in reel.cuts[1:]]

    assert positions == sorted(positions)
    assert len(set(positions)) == len(positions), "a clip was reused unnecessarily"


def test_running_out_of_footage_wraps_to_the_start(clips):
    """With more shots than clips the order cannot stay forward for ever. It
    goes back to the beginning and runs forward again — a second pass — rather
    than jumping about, which would read as a shuffle."""
    reel = plan_reel(clips, duration=45, pace="intense", log_fn=lambda *_: None)
    positions = [clips.index(c.source) for c in reel.cuts[1:]]

    wraps = sum(1 for a, b in zip(positions, positions[1:]) if b < a)
    passes = len(positions) / len(clips)

    assert wraps <= passes + 1, "the order jumps around rather than wrapping"


# ---------------------------------------------------------------------------
# Music
# ---------------------------------------------------------------------------

def _off_beat(duration: float, beat: float) -> float:
    """How far a duration is from the nearest whole number of beats.

    Not ``duration % beat``: a cut a hair *under* a multiple — which is what
    ``end - start`` gives whenever the in-point is not itself on the grid —
    comes back from the modulo as very nearly a whole beat rather than as very
    nearly zero, and reads as maximally off the beat when it is as on it as a
    float can be.
    """
    remainder = duration % beat
    return min(remainder, beat - remainder)


def test_shots_land_on_the_beat_when_there_is_music(clips):
    beat = 0.5   # 120 BPM
    reel = plan_reel(clips, duration=24, pace="energetic",
                     analysis=_Analysis(120), log_fn=lambda *_: None)

    for cut in reel.cuts:
        assert _off_beat(cut.duration, beat) == pytest.approx(0.0, abs=0.02)


def test_shots_still_land_on_the_beat_with_the_in_points_moved(clips):
    """Settling starts a shot partway into its clip, which takes the in-point
    off the musical grid even though the length stays on it. The length is
    what the ear hears, so it is the length that has to survive."""
    beat = 0.5
    graded = {p: _windows(p, [0.05] * 16 + [0.95] * 32) for p in clips}

    reel = plan_reel(clips, duration=24, pace="energetic", classify=False,
                     analysis=_Analysis(120), windows=graded,
                     log_fn=lambda *_: None)

    assert any(c.start > 0.25 for c in reel.cuts), "nothing moved, so this proves nothing"
    for cut in reel.cuts:
        assert _off_beat(cut.duration, beat) == pytest.approx(0.0, abs=0.02)


def test_music_does_not_flatten_the_paces_together(clips):
    """Snapping to bars would round a 1.75s energetic shot and a 3s vlog shot
    to the same 2s, and the preset would stop meaning anything. The beat is
    fine enough to keep them apart."""
    analysis = _Analysis(120)
    lengths = {}
    for pace in ("vlog", "energetic", "intense"):
        reel = plan_reel(clips, duration=45, pace=pace, analysis=analysis,
                         log_fn=lambda *_: None)
        body = [c.duration for c in reel.cuts if c.label == "Escalation"]
        lengths[pace] = body[0]

    assert lengths["intense"] < lengths["energetic"] < lengths["vlog"]


def test_no_music_still_produces_a_reel(clips):
    reel = plan_reel(clips, duration=24, pace="energetic", analysis=None,
                     log_fn=lambda *_: None)

    assert reel.cuts
    assert reel.duration == pytest.approx(24, rel=0.15)


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------

def test_hook_text_lands_on_the_first_shot_only(clips):
    """Most viewers start muted, so the opening line is part of the edit — but
    repeating it on every shot of the section would be a wall of text."""
    reel = plan_reel(clips, duration=24, pace="energetic",
                     texts={"Hook": "I nearly quit at mile 38",
                            "Payoff": "The answer was slowing down"},
                     log_fn=lambda *_: None)

    assert reel.cuts[0].text == "I nearly quit at mile 38"
    assert sum(1 for c in reel.cuts if c.text) == 2
    assert [c.text for c in reel.cuts if c.label == "Payoff"][0].startswith("The answer")


# ---------------------------------------------------------------------------
# Awkward input
# ---------------------------------------------------------------------------

def test_fewer_clips_than_shots_reuses_them_rather_than_coming_up_short(tmp_path):
    """A 24-second reel wants a dozen shots and a short shoot may have four."""
    few = []
    for i in range(3):
        path = tmp_path / f"c{i}.mp4"
        subprocess.run(
            [FFMPEG, "-y", "-v", "error", "-f", "lavfi",
             "-i", "color=c=gray:size=64x48:rate=30:duration=10",
             "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
             "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
             str(path)], check=True, capture_output=True)
        few.append(str(path))

    reel = plan_reel(few, duration=24, pace="energetic", log_fn=lambda *_: None)

    assert len(reel.cuts) > len(few)
    assert reel.duration == pytest.approx(24, rel=0.2)


def test_slices_of_one_source_do_not_overlap(tmp_path):
    """Reusing a clip means taking the *next* piece of it, not the same piece
    again — otherwise the reel repeats itself."""
    path = str(tmp_path / "only.mp4")
    subprocess.run(
        [FFMPEG, "-y", "-v", "error", "-f", "lavfi",
         "-i", "color=c=gray:size=64x48:rate=30:duration=30",
         "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
         "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
         path], check=True, capture_output=True)

    reel = plan_reel([path], duration=20, pace="energetic", log_fn=lambda *_: None)

    spans = sorted((c.start, c.end) for c in reel.cuts)
    for (_, end), (next_start, _) in zip(spans, spans[1:]):
        assert next_start >= end - 0.01


def test_an_impossible_target_says_so(clips):
    """A calm 12-second reel cannot hold a seven-shot structure. Returning
    something half again as long without a word is the unhelpful option."""
    logged: list[str] = []

    plan_reel(clips, duration=12, pace="calm", log_fn=logged.append)

    assert any("shorter than" in m for m in logged)
    assert any("energetic" in m.lower() or "vlog" in m.lower() for m in logged)


def test_minimum_duration_is_reported_before_a_render(clips):
    assert minimum_duration("calm") > minimum_duration("intense")
    assert minimum_duration("intense") < 12.0


def test_no_clips_is_refused():
    with pytest.raises(ValueError, match="no usable clips"):
        plan_reel([], duration=24, log_fn=lambda *_: None)


def test_an_unknown_pace_is_refused(clips):
    with pytest.raises(ValueError, match="unknown pace"):
        plan_reel(clips, duration=24, pace="brisk", log_fn=lambda *_: None)


def test_clips_shorter_than_a_shot_are_ignored(tmp_path, clips):
    """A tenth-of-a-second file is not a shot and would only produce a flash."""
    tiny = str(tmp_path / "tiny.mp4")
    subprocess.run(
        [FFMPEG, "-y", "-v", "error", "-f", "lavfi",
         "-i", "color=c=red:size=64x48:rate=30:duration=0.2",
         "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
         "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
         tiny], check=True, capture_output=True)

    reel = plan_reel([tiny] + clips, duration=24, pace="energetic",
                     log_fn=lambda *_: None)

    assert all(c.source != tiny for c in reel.cuts)


def test_no_shot_is_shorter_than_the_floor(clips):
    reel = plan_reel(clips, duration=12, pace="intense", log_fn=lambda *_: None)

    assert all(c.duration >= MIN_SHOT - 0.01 for c in reel.cuts)


# ---------------------------------------------------------------------------
# Presentation
# ---------------------------------------------------------------------------

def test_the_plan_describes_itself_by_section(clips):
    """A dozen near-identical lines is the output nobody reads."""
    reel = plan_reel(clips, duration=24, pace="energetic", log_fn=lambda *_: None)

    text = describe_plan(reel)

    assert "cuts/min" in text
    for section in STRUCTURE:
        assert section.name in text
    assert len(text.splitlines()) == len(STRUCTURE) + 1


def test_the_offered_lengths_are_the_ones_worth_testing():
    assert [seconds for seconds, _ in LENGTHS] == [15, 24, 50]
    assert all(reason for _, reason in LENGTHS)


def test_each_pace_reports_the_band_it_claims():
    for pace in PACES.values():
        low, high = pace.cuts_per_minute
        assert low < high
        assert (low, high) == (int(60 / pace.max_shot), int(60 / pace.min_shot))


# ---------------------------------------------------------------------------
# Where inside a clip a shot starts
# ---------------------------------------------------------------------------
#
# Before this, every first slice of a clip began at frame zero — which on real
# footage is very often the camera still being raised, swung round or pulled
# out of a pocket. Windows are injected here rather than measured, because the
# measurement needs OpenCV (mocked in this suite) and what the planner owes is
# only that it *uses* what it is given.

def _windows(path, qualities, rate=8.0):
    from modules.shot_window import ClipWindows, Sample

    return ClipWindows(
        path=path, duration=len(qualities) / rate, measured=True,
        samples=[Sample(t=i / rate, usable=q) for i, q in enumerate(qualities)])


def test_a_shot_starts_where_the_camera_settled(clips):
    """The whole point: a clip whose first two seconds are the camera being
    placed must be cut from later on, not from the top."""
    graded = {p: _windows(p, [0.05] * 16 + [0.95] * 32) for p in clips}

    reel = plan_reel(clips, duration=24, pace="energetic", classify=False,
                     windows=graded, log_fn=lambda *_: None)

    assert all(c.start >= 1.5 for c in reel.cuts), \
        [round(c.start, 2) for c in reel.cuts]


def test_a_clip_that_starts_well_is_still_cut_from_the_top(clips):
    """Moving in-points on principle would be its own bug."""
    graded = {p: _windows(p, [0.9] * 48) for p in clips}

    reel = plan_reel(clips, duration=24, pace="energetic", classify=False,
                     windows=graded, log_fn=lambda *_: None)

    assert min(c.start for c in reel.cuts) == pytest.approx(0.0)


def test_settling_can_be_turned_off(clips):
    """A caller that cannot afford the measurement, or does not want it, gets
    exactly the old behaviour."""
    graded = {p: _windows(p, [0.05] * 16 + [0.95] * 32) for p in clips}

    reel = plan_reel(clips, duration=24, pace="energetic", classify=False,
                     windows=graded, settle=False, log_fn=lambda *_: None)

    assert min(c.start for c in reel.cuts) == pytest.approx(0.0)


def test_settling_still_does_not_overlap_slices_of_one_source(tmp_path):
    """Searching for the best window per shot must stay forward-only, or a
    reused clip shows the same seconds twice."""
    path = str(tmp_path / "only.mp4")
    subprocess.run(
        [FFMPEG, "-y", "-v", "error", "-f", "lavfi",
         "-i", "color=c=gray:size=64x48:rate=30:duration=30",
         "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
         "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
         path], check=True, capture_output=True)
    # The best second of the clip is right at the top, so a planner that
    # searched the whole clip every time would keep returning it.
    graded = {path: _windows(path, [0.99] * 8 + [0.4] * 232)}

    reel = plan_reel([path], duration=20, pace="energetic", classify=False,
                     windows=graded, log_fn=lambda *_: None)

    spans = sorted((c.start, c.end) for c in reel.cuts)
    for (_, end), (next_start, _) in zip(spans, spans[1:]):
        assert next_start >= end - 0.01


def test_settling_keeps_shooting_order(clips):
    """An in-point moves within a clip; it never reorders the clips. The
    progression is what makes a sequence read as a story, and trading it for a
    cleaner frame is a bad bargain."""
    graded = {p: _windows(p, [0.1] * 8 + [0.9] * 40) for p in clips}
    # Make one late clip conspicuously the best, so a planner that ranked on
    # quality would pull it forward.
    graded[clips[-1]] = _windows(clips[-1], [0.99] * 48)

    reel = plan_reel(clips, duration=24, pace="energetic", classify=False,
                     windows=graded, log_fn=lambda *_: None)

    body = [c.source for c in reel.cuts if c.label in ("Context", "Escalation")]
    positions = [clips.index(s) for s in body]

    # Forward, except where it runs out of footage and wraps to the start —
    # the same allowance test_running_out_of_footage_wraps_to_the_start makes.
    wraps = sum(1 for a, b in zip(positions, positions[1:]) if b < a)
    assert wraps <= 1, positions
    # And the conspicuously best clip did not get dragged to the front, which
    # is what ranking the body on picture quality would have done.
    assert positions[0] < len(clips) - 1, positions


def test_a_moved_in_point_is_reported(clips):
    """An in-point the user did not choose looks like a bug until something
    says why it happened."""
    logged: list[str] = []
    graded = {p: _windows(p, [0.05] * 16 + [0.95] * 32) for p in clips}

    plan_reel(clips, duration=24, pace="energetic", classify=False,
              windows=graded, log_fn=logged.append)

    assert any("start later than frame zero" in line for line in logged)


def test_an_unmeasured_clip_falls_back_to_the_top(clips):
    """Measuring failing costs the improvement, never the reel."""
    from modules.shot_window import ClipWindows

    graded = {p: ClipWindows(path=p, duration=6.0) for p in clips}

    reel = plan_reel(clips, duration=24, pace="energetic", classify=False,
                     windows=graded, log_fn=lambda *_: None)

    assert min(c.start for c in reel.cuts) == pytest.approx(0.0)
    assert len(reel.cuts) > 1


def test_the_feather_reaches_every_cut(clips):
    reel = plan_reel(clips, duration=24, pace="energetic", classify=False,
                     transition="iris_open", feather=0.3, settle=False,
                     log_fn=lambda *_: None)

    assert all(c.feather == pytest.approx(0.3) for c in reel.cuts)


# ---------------------------------------------------------------------------
# Not showing the same thing twice
# ---------------------------------------------------------------------------
#
# Places are injected rather than measured: reading them needs ffprobe against
# real camera files, and what the planner owes is only that it acts on what it
# is given. modules/test_shot_place.py pins the reading.

def test_the_reel_visits_every_spot_before_repeating_one(clips):
    """Four clips from one spot and eight from their own must not put four
    shots of the same view in a twelve-shot reel."""
    places = {p: (0 if i < 4 else i) for i, p in enumerate(clips)}

    reel = plan_reel(clips, duration=24, pace="energetic", classify=False,
                     settle=False, places=places, log_fn=lambda *_: None)

    from collections import Counter
    used = Counter(places[c.source] for c in reel.cuts)
    crowded = used[0]
    assert crowded <= 2, f"the busy spot supplied {crowded} of {len(reel.cuts)} shots"


def test_two_shots_from_one_spot_are_never_side_by_side(clips):
    """With fewer places than shots something has to repeat. Repeating a view
    later is a montage; repeating it immediately is a mistake."""
    places = {p: i % 3 for i, p in enumerate(clips)}

    reel = plan_reel(clips, duration=45, pace="intense", classify=False,
                     settle=False, places=places, log_fn=lambda *_: None)

    adjacent = [(a, b) for a, b in zip(reel.cuts, reel.cuts[1:])
                if places[a.source] == places[b.source]]
    assert not adjacent, f"{len(adjacent)} pair(s) of shots share a spot"


def test_spreading_relaxes_rather_than_running_short(clips):
    """Everything from one spot is still a reel — the preference gives way
    rather than refusing to cut."""
    places = {p: 0 for p in clips}

    reel = plan_reel(clips, duration=24, pace="energetic", classify=False,
                     settle=False, places=places, log_fn=lambda *_: None)

    assert len(reel.cuts) > 5
    assert reel.duration == pytest.approx(24, rel=0.2)


def test_spreading_can_be_turned_off(clips):
    off = plan_reel(clips, duration=24, pace="energetic", classify=False,
                    settle=False, spread=False, log_fn=lambda *_: None)

    assert len(off.cuts) > 1


def test_a_repeated_spot_is_reported(clips):
    """A view shown twice is what the viewer notices, so the log says so
    either way rather than leaving it to be discovered on playback."""
    logged: list[str] = []
    plan_reel(clips, duration=45, pace="intense", classify=False, settle=False,
              places={p: i % 2 for i, p in enumerate(clips)},
              log_fn=logged.append)

    assert any("shown twice" in line for line in logged)

    logged.clear()
    plan_reel(clips, duration=24, pace="energetic", classify=False, settle=False,
              places={p: i for i, p in enumerate(clips)}, log_fn=logged.append)
    assert any("different spot" in line for line in logged)


def test_spreading_costs_a_sweep_rather_than_the_order(clips):
    """Spreading and strict shooting order genuinely pull against each other,
    and it is worth being precise about what is given up.

    With two clips per spot, visiting every spot before repeating one means
    taking the first clip of each place, then coming back for the second. So
    the body is no longer one forward run — it is two, each forward. That
    reads as a montage that returns to where it has been, which is a montage;
    what it must never become is a shuffle, so each sweep still runs forward
    and there are no more sweeps than there are clips sharing a spot.
    """
    per_place = 2
    places = {p: i // per_place for i, p in enumerate(clips)}

    reel = plan_reel(clips, duration=24, pace="energetic", classify=False,
                     settle=False, places=places, log_fn=lambda *_: None)

    body = [clips.index(c.source) for c in reel.cuts
            if c.label in ("Context", "Escalation")]
    wraps = sum(1 for a, b in zip(body, body[1:]) if b < a)

    assert wraps <= per_place, body
    # Within a sweep the order is forward — no jumping backwards twice in a row.
    runs, current = [], [body[0]]
    for previous, position in zip(body, body[1:]):
        if position < previous:
            runs.append(current)
            current = []
        current.append(position)
    runs.append(current)
    for run in runs:
        assert run == sorted(run), f"a sweep runs backwards: {run}"
