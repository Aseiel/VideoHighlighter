"""Tests for `modules.music_analysis`.

Fixtures are synthesised with numpy and the stdlib `wave` module, so the
default path of this file needs no ffmpeg, no network and no music: a click
track is a signal whose beat times are known exactly, which is the only way to
say whether a beat tracker is right rather than merely plausible.

The trap this file exists to catch is a beat grid that *looks* fine — evenly
spaced, sensible count — while sitting at half or double the real tempo, or
drifting a few milliseconds per bar until the last cut lands off the beat.
Both produce an edit that feels wrong for reasons nobody can name, and neither
shows up unless the test knows where the beats really were.

Only the numpy backend is exercised. librosa is optional, is not installed in
the test environment, and is mocked out by tests/conftest.py — which is itself
a trap worth a test, since a mock imports perfectly and answers every question
with another mock.
"""

from __future__ import annotations

import math
import subprocess
import wave

import numpy as np
import pytest

from modules.app_paths import ffmpeg_exe
from modules import music_analysis as ma
from modules.music_analysis import (
    MusicAnalysis,
    Section,
    analyze_music,
    beat_aligned_durations,
    load_analysis,
    save_analysis,
    snap_segments,
    snap_to_beat,
)

SR = 22050

# One STFT hop is 23.2 ms, so a beat cannot be located finer than that; the
# envelope also peaks on the rising edge of the analysis window rather than at
# its centre, which puts detections a few ms early. 40 ms covers both and is
# still well inside the ~50 ms at which a cut stops reading as on the beat.
BEAT_TOLERANCE = 0.040

# Tempo is held to a third of a BPM, which is far tighter than anyone can hear
# in isolation and is the point: BPM error is *drift*. A caller extrapolating
# the grid across a three-minute track turns 0.3 BPM into nearly half a second
# by the end, and the last cut lands visibly late. Measured error on these
# fixtures is under 0.1 BPM, so this is a three-fold margin, not a fitted
# threshold.
BPM_TOLERANCE = 0.3


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------
def _write_wav(path, samples, sample_rate=SR, channels=1) -> str:
    """16-bit PCM wav from float samples in [-1, 1]."""
    data = np.clip(np.asarray(samples, dtype=np.float64), -1.0, 1.0)
    pcm = (data * 20000.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    return str(path)


def _click_track(bpm, duration, *, accent_every=0, meter=4, seed=0,
                 gain=1.0, sample_rate=SR):
    """Percussive clicks at an exact tempo, plus the true click times.

    The click is a 10 ms burst of decaying noise rather than a sine pip:
    spectral flux measures broadband *rise*, and a tone burst excites so few
    bins that it under-reports what a drum does to the envelope.
    """
    rng = np.random.default_rng(seed)
    signal = np.zeros(int(duration * sample_rate), dtype=np.float64)
    length = int(0.010 * sample_rate)
    burst = rng.standard_normal(length) * np.exp(-np.linspace(0.0, 8.0, length))
    burst /= np.abs(burst).max()

    period = 60.0 / bpm
    times = []
    index = 0
    position = 0.0
    while position * sample_rate + length < signal.size:
        start = int(round(position * sample_rate))
        loud = accent_every and index % accent_every == 0
        signal[start:start + length] += burst * (1.8 if loud else 1.0) * gain
        times.append(position)
        position += period
        index += 1
    return signal, times


def _worst_alignment(beats, click_times):
    """Largest distance from a true click to the nearest detected beat.

    The click at t=0 is excluded: the onset envelope has no frame before the
    start of the file, so an attack on sample zero produces no rise for the
    flux to see. Real music does not start on sample zero, and the grid locks
    on from the second click regardless — but a test that ignored this would
    be asserting against an artifact of its own fixture.
    """
    grid = np.asarray(beats, dtype=np.float64)
    assert grid.size, "no beats detected at all"
    return max(float(np.min(np.abs(grid - t))) for t in click_times[1:])


@pytest.fixture(scope="module")
def click_120(tmp_path_factory):
    """30 s of clicks every 0.5 s — 120 BPM, exactly."""
    root = tmp_path_factory.mktemp("music_analysis")
    signal, times = _click_track(120.0, 30.0)
    return _write_wav(root / "click120.wav", signal), times


def _silent_analysis(beats, *, meter=4, bpm=120.0):
    """A MusicAnalysis with a hand-built grid, for testing the snap helpers.

    Built directly rather than analysed: these functions are pure geometry over
    a list of beat times, and running a decoder to obtain that list would test
    the decoder instead.
    """
    interval = 60.0 / bpm if bpm > 0 else 0.0
    return MusicAnalysis(
        path="synthetic",
        duration=(beats[-1] if beats else 0.0),
        sample_rate=SR,
        bpm=bpm,
        beats=list(beats),
        downbeats=list(beats[::meter]),
        beat_interval=interval,
        meter=meter,
        onset_envelope=[],
        onset_times=[],
        sections=[],
        backend="numpy",
    )


# ---------------------------------------------------------------------------
# The correctness bar: does it find the beat that is actually there
# ---------------------------------------------------------------------------
def test_click_track_tempo_and_beats(click_120):
    """A 120 BPM click track must come back as 120 BPM with beats on the clicks.

    This is the whole point of the module. A tracker can report a believable
    tempo and still place the grid between the beats, so the tempo and the beat
    positions are both checked against ground truth the fixture knows.
    """
    path, clicks = click_120
    analysis = analyze_music(path, backend="numpy", log_fn=lambda *_: None)

    assert analysis.backend == "numpy"
    assert abs(analysis.bpm - 120.0) < BPM_TOLERANCE
    assert analysis.beat_interval == pytest.approx(60.0 / analysis.bpm)
    # 60 clicks in 30 s; the first is unobservable (see _worst_alignment).
    assert len(analysis.beats) >= 58
    assert _worst_alignment(analysis.beats, clicks) < BEAT_TOLERANCE
    assert analysis.beats == sorted(analysis.beats)


@pytest.mark.parametrize("bpm", [75.0, 100.0, 140.0, 160.0, 168.0, 200.0])
def test_tempo_is_not_halved_or_doubled(tmp_path, bpm):
    """Tempi either side of the 120 BPM prior must not be pulled to it.

    Autocorrelation scores half and double a tempo almost as well as the tempo,
    so the prior that resolves the tie is also the thing most likely to impose
    an answer. 75 and 160 BPM are the cases that would break: both are an
    obvious harmonic hop away from something nearer 120.

    168 and 200 are the far end of the same problem, and the end that actually
    broke. A tempo and its half sit equally far from the centre of the prior
    when their geometric mean is 120 BPM — at 170 — so from there upward the
    prior stops leaning toward the true tempo and starts leaning against it. At
    200 it prefers 100 by a quarter, while the correlation that is supposed to
    outvote it separates the two by about a percent.
    """
    signal, clicks = _click_track(bpm, 24.0)
    path = _write_wav(tmp_path / f"click{int(bpm)}.wav", signal)
    analysis = analyze_music(path, backend="numpy", log_fn=lambda *_: None)

    assert abs(analysis.bpm - bpm) < BPM_TOLERANCE
    assert _worst_alignment(analysis.beats, clicks) < BEAT_TOLERANCE


@pytest.mark.parametrize("bpm", [156.0, 172.0, 180.0, 190.0, 196.0])
def test_the_fast_end_of_the_search_range_is_reachable(tmp_path, bpm):
    """Every tempo under MAX_BPM has to be an answer the module can give.

    Fourteen of the twenty-six even tempi between 150 and 200 used to come back
    at exactly half — all of them above 190 — on click tracks carrying no
    half-tempo cue whatsoever: every click identical, no accents, and the same
    result for every seed and every fixture length. The tempo is only the
    headline. Half the beats are emitted too, so half the clicks have nothing
    within 50 ms of them, and `beat_aligned_durations` hands back a bar twice
    as long as the music's — a cutter asking for two bars silently gets four.
    """
    signal, clicks = _click_track(bpm, 24.0)
    path = _write_wav(tmp_path / f"fast{int(bpm)}.wav", signal)
    analysis = analyze_music(path, backend="numpy", log_fn=lambda *_: None)

    assert abs(analysis.bpm - bpm) < BPM_TOLERANCE
    assert _worst_alignment(analysis.beats, clicks) < BEAT_TOLERANCE
    # Half tempo halves the beat count too, which is the part a caller feels.
    assert len(analysis.beats) >= len(clicks) - 3


def test_an_accent_pattern_is_not_read_as_a_half_tempo_cue(tmp_path):
    """A bar's accents must not make the beats between them look optional.

    What separates a period from its double is whether the midpoints between
    the candidate beats carry beats of their own, and the trap is measuring
    that against the average beat: at twice the real period one of the two
    interleaved grids swallows every accent in the bar, and averages higher
    than the other for that reason alone. One 4/4 accent on a 168 BPM track is
    enough to bring the halving straight back if each midpoint is not compared
    against the weaker of the two beats it sits between.
    """
    signal, clicks = _click_track(168.0, 24.0, accent_every=4)
    path = _write_wav(tmp_path / "accented_fast.wav", signal)
    analysis = analyze_music(path, backend="numpy", log_fn=lambda *_: None)

    assert abs(analysis.bpm - 168.0) < BPM_TOLERANCE
    assert _worst_alignment(analysis.beats, clicks) < BEAT_TOLERANCE


def test_a_real_half_tempo_cue_is_still_obeyed(tmp_path):
    """Deciding the octave on evidence must not become a bias toward the fast
    reading.

    Loud and soft attacks alternating is what a track whose beat is half the
    attack rate actually sounds like, and there the slower grid is the right
    answer: 180 attacks a minute with every other one accented is 90 BPM, not
    180. A rule loose enough to call the soft attacks beats would double the
    tempo of any music with a subdivision in it — which is most music — and
    the click-track tests above would not notice.
    """
    signal, clicks = _click_track(180.0, 24.0, accent_every=2)
    path = _write_wav(tmp_path / "backbeat.wav", signal)
    analysis = analyze_music(path, backend="numpy", log_fn=lambda *_: None)

    assert abs(analysis.bpm - 90.0) < BPM_TOLERANCE
    # Every beat on an accent, not merely the right number of them: a grid at
    # 90 BPM in the wrong phase sits on the soft attacks and looks identical
    # in the tempo alone.
    accents = clicks[::2]
    assert analysis.beats
    for beat in analysis.beats:
        assert min(abs(beat - t) for t in accents) < BEAT_TOLERANCE


def test_downbeats_are_a_subset_on_the_accented_phase(tmp_path):
    """Downbeats must fall on the accent, not merely every fourth beat.

    Picking beats[0::meter] without choosing a phase is right one time in four
    and looks correct in every debug print, because the spacing is right
    either way.
    """
    signal, clicks = _click_track(120.0, 24.0, accent_every=4)
    path = _write_wav(tmp_path / "accented.wav", signal)
    analysis = analyze_music(path, backend="numpy", meter=4, log_fn=lambda *_: None)

    assert analysis.downbeats, "no downbeats found on an accented track"
    beat_set = {round(b, 9) for b in analysis.beats}
    assert all(round(d, 9) in beat_set for d in analysis.downbeats)

    gaps = np.diff(analysis.downbeats)
    assert np.allclose(gaps, 4 * analysis.beat_interval, atol=BEAT_TOLERANCE)
    # Accents sit on the even seconds (every 4th click at 2 clicks/s).
    for downbeat in analysis.downbeats:
        assert min(abs(downbeat - t) for t in clicks[::4]) < BEAT_TOLERANCE


def test_meter_three_takes_every_third_beat(tmp_path):
    """meter is a parameter, not a constant 4 baked into the downbeat step."""
    signal, _ = _click_track(120.0, 18.0, accent_every=3)
    path = _write_wav(tmp_path / "waltz.wav", signal)
    analysis = analyze_music(path, backend="numpy", meter=3, log_fn=lambda *_: None)

    assert analysis.meter == 3
    gaps = np.diff(analysis.downbeats)
    assert np.allclose(gaps, 3 * analysis.beat_interval, atol=BEAT_TOLERANCE)


def test_stereo_and_off_rate_wav_is_resampled(tmp_path):
    """A 44.1 kHz stereo file must analyse to the same tempo as its mono twin.

    Channel folding and resampling both happen before any analysis, so getting
    either wrong shifts every frame index — and the failure looks like a tempo
    error rather than a decode error.
    """
    signal, clicks = _click_track(120.0, 16.0, sample_rate=44100)
    interleaved = np.stack([signal, signal * 0.5], axis=1).reshape(-1)
    path = _write_wav(tmp_path / "stereo.wav", interleaved,
                      sample_rate=44100, channels=2)
    analysis = analyze_music(path, backend="numpy", log_fn=lambda *_: None)

    assert analysis.sample_rate == ma.TARGET_SR
    assert analysis.duration == pytest.approx(16.0, abs=0.05)
    assert abs(analysis.bpm - 120.0) < BPM_TOLERANCE
    assert _worst_alignment(analysis.beats, clicks) < BEAT_TOLERANCE


def test_readable_wav_never_spawns_ffmpeg(click_120, monkeypatch):
    """A PCM wav is read in-process; the ffmpeg detour is a fallback, not a step.

    Skipping the subprocess is both a speed win and what keeps the module
    usable where ffmpeg is missing, so it needs a test that fails loudly if
    someone routes every input through the decoder again.
    """
    path, _ = click_120

    def _explode():
        raise AssertionError("ffmpeg was invoked for a readable PCM wav")

    monkeypatch.setattr(ma, "ffmpeg_exe", _explode)
    analysis = analyze_music(path, backend="numpy", log_fn=lambda *_: None)
    assert abs(analysis.bpm - 120.0) < BPM_TOLERANCE


# ---------------------------------------------------------------------------
# Degenerate input: the cases that divide by zero
# ---------------------------------------------------------------------------
def test_silence_reports_no_tempo_instead_of_guessing(tmp_path):
    """Silence must yield an empty grid, not a default tempo and phantom beats.

    Every downstream helper divides by the beat interval, so the temptation is
    to hand back 120 BPM and keep going. That produces cut points spaced to
    music that is not playing, which is worse than not snapping at all.
    """
    path = _write_wav(tmp_path / "silence.wav", np.zeros(2 * SR))
    logged = []
    analysis = analyze_music(path, backend="numpy", log_fn=logged.append)

    assert analysis.bpm == 0.0
    assert analysis.beats == []
    assert analysis.downbeats == []
    assert analysis.beat_interval == 0.0
    assert analysis.has_beats is False
    assert beat_aligned_durations(analysis, bars=2) == 0.0
    assert snap_to_beat(1.234, analysis) == 1.234
    assert any("⚠️" in line for line in logged), "silence should warn the user"
    # Sections still describe the file, and silence is the bottom tier.
    assert [s.label for s in analysis.sections] == ["low"]


@pytest.mark.parametrize("samples", [0, 1, int(0.05 * SR), ma.N_FFT])
def test_very_short_files_do_not_crash(tmp_path, samples):
    """Files shorter than one FFT window must return an empty grid, not raise.

    A zero-sample file makes every normalisation a division by zero and every
    frame count negative; users hand this module whatever is on disk, including
    a truncated download.
    """
    path = _write_wav(tmp_path / f"short{samples}.wav", np.zeros(samples))
    analysis = analyze_music(path, backend="numpy", log_fn=lambda *_: None)

    assert analysis.bpm == 0.0
    assert analysis.beats == []
    assert analysis.duration == pytest.approx(samples / SR, abs=1e-6)
    assert snap_segments([(0.0, 1.0)], analysis) == [(0.0, 1.0)]


def test_a_single_beat_is_not_a_grid(tmp_path):
    """Two attacks in a file are not a tempo and must not be reported as one.

    The autocorrelation names a lag for anything — it is an argmax over a range
    that excludes nothing — so a file the tracker gets one beat out of still
    comes back with a confident BPM hung off it. Every snap then lands on that
    single instant, so a caller that correctly gates on `has_beats` stacks the
    whole reel at one moment, which is strictly worse than the no-rhythm path
    that leaves the times where the selector put them.
    """
    signal = np.zeros(3 * SR)
    signal[SR] = 1.0
    signal[2 * SR] = 1.0
    path = _write_wav(tmp_path / "two_impulses.wav", signal)

    analysis = analyze_music(path, backend="numpy", log_fn=lambda *_: None)

    assert analysis.bpm == 0.0
    assert analysis.beats == []
    assert analysis.beat_interval == 0.0
    assert analysis.has_beats is False
    assert beat_aligned_durations(analysis) == 0.0
    assert snap_to_beat(0.2, analysis) == 0.2
    assert snap_to_beat(2.9, analysis) == 2.9

    # The flag has to refuse a one-beat grid, not merely never be handed one:
    # `load_analysis` will rebuild exactly that from a sidecar written earlier.
    assert _silent_analysis([1.997]).has_beats is False


def test_argument_errors_raise(tmp_path, click_120):
    """Wrong arguments are the caller's bug and must not be papered over."""
    path, _ = click_120
    with pytest.raises(ValueError):
        analyze_music(str(tmp_path / "missing.wav"), backend="numpy")
    with pytest.raises(ValueError):
        analyze_music(path, backend="tensorflow")
    with pytest.raises(ValueError):
        analyze_music(path, meter=0, log_fn=lambda *_: None)


def test_mocked_librosa_is_not_treated_as_real():
    """conftest installs a MagicMock as `librosa`; auto must still pick numpy.

    A mock satisfies `import librosa` and then returns a MagicMock for every
    array it is asked for, which would flow through the whole analysis and
    produce a result made of nothing that no assertion here would catch.
    """
    import librosa  # the conftest mock, in this environment

    if isinstance(getattr(librosa, "__version__", None), str):
        pytest.skip("a real librosa is installed; nothing to guard against")
    assert ma._real_librosa() is None


def test_explicit_librosa_request_degrades_to_numpy(click_120):
    """Asking for a backend that is not installed logs and continues.

    The analysis is the deliverable; which library produced it is not. The
    `backend` field is what tells the caller what actually ran.
    """
    path, _ = click_120
    if ma._real_librosa() is not None:
        pytest.skip("librosa is installed; the fallback cannot be exercised")

    logged = []
    analysis = analyze_music(path, backend="librosa", log_fn=logged.append)
    assert analysis.backend == "numpy"
    assert abs(analysis.bpm - 120.0) < BPM_TOLERANCE
    assert any("librosa" in line for line in logged)


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------
def test_sections_follow_energy_and_stay_long_enough(tmp_path):
    """Energy tiers must track the arrangement, and no section may be a blip.

    The merge rule is the part that matters: without it a one-second fill
    becomes its own section, and a cutter reacting to section changes produces
    exactly the twitchiness the sections exist to prevent.
    """
    quiet, _ = _click_track(120.0, 9.0, gain=0.15, seed=1)
    loud, _ = _click_track(120.0, 9.0, gain=1.0, seed=2)
    blip, _ = _click_track(120.0, 1.0, gain=1.0, seed=3)
    tail, _ = _click_track(120.0, 9.0, gain=0.3, seed=4)
    signal = np.concatenate([quiet, loud, blip, tail])
    path = _write_wav(tmp_path / "arrangement.wav", signal)

    analysis = analyze_music(path, backend="numpy", log_fn=lambda *_: None)
    sections = analysis.sections

    assert len(sections) >= 2
    assert sections[0].start == 0.0
    assert sections[-1].end == pytest.approx(analysis.duration, abs=1e-6)
    for earlier, later in zip(sections, sections[1:]):
        assert earlier.end == pytest.approx(later.start, abs=1e-6)
    for section in sections:
        assert section.label in ma.ENERGY_TIERS
        assert 0.0 <= section.energy <= 1.0
        assert section.duration >= ma.MIN_SECTION - 1e-6

    # The quiet opening must not be filed at the same tier as the loud middle.
    assert sections[0].label != max(sections, key=lambda s: s.energy).label


def test_a_file_that_is_not_whole_seconds_has_no_short_final_section(tmp_path):
    """The minimum length is seconds of audio, and windows are not seconds.

    Windows are a second wide and the last one is zero-padded out to a full
    one, so a merge rule that counts windows credits that padding as music: a
    9.1 s file ends on a run of four windows holding 3.1 s of audio, which
    passes a four-window minimum while failing the four-second minimum the
    windows stand for. The fixture above is exactly 28.000 s and cannot catch
    this; a music bed that is a whole number of seconds is the rarity, not the
    rule.
    """
    quiet, _ = _click_track(120.0, 6.0, gain=0.15, seed=1)
    loud, _ = _click_track(120.0, 3.1, gain=1.0, seed=2)
    path = _write_wav(tmp_path / "ragged.wav", np.concatenate([quiet, loud]))

    analysis = analyze_music(path, backend="numpy", log_fn=lambda *_: None)

    # The fixture only bites while its tail is a part-window, so pin that.
    assert 0.05 < analysis.duration % 1.0 < 0.95
    assert analysis.sections
    for section in analysis.sections:
        assert section.duration >= ma.MIN_SECTION - 1e-9
    assert analysis.sections[0].start == 0.0
    assert analysis.sections[-1].end == pytest.approx(analysis.duration, abs=1e-6)


# ---------------------------------------------------------------------------
# Snapping
# ---------------------------------------------------------------------------
def test_uniform_energy_still_lands_in_the_right_tier(click_120, tmp_path):
    """A track with no dynamics must not be filed by an arbitrary tercile.

    Terciles of a constant are all the same number, so every window sits on the
    boundary and the comparison alone decides the tier — which would file a
    wall-to-wall loud track as "low" purely because `<=` was written rather
    than `<`. Both ends of the scale are checked, since a fallback that gets
    silence right by accident is not a fallback.
    """
    loud_path, _ = click_120
    loud = analyze_music(loud_path, backend="numpy", log_fn=lambda *_: None)
    assert [s.label for s in loud.sections] == ["high"]

    quiet_path = _write_wav(tmp_path / "quiet.wav", np.zeros(5 * SR))
    quiet = analyze_music(quiet_path, backend="numpy", log_fn=lambda *_: None)
    assert [s.label for s in quiet.sections] == ["low"]


def test_snap_modes_pick_the_right_neighbour():
    """nearest/previous/next must differ, and tie-breaks must be stable."""
    analysis = _silent_analysis([0.0, 0.5, 1.0, 1.5, 2.0])

    assert snap_to_beat(1.1, analysis, mode="nearest") == 1.0
    assert snap_to_beat(1.4, analysis, mode="nearest") == 1.5
    assert snap_to_beat(1.1, analysis, mode="previous") == 1.0
    assert snap_to_beat(1.1, analysis, mode="next") == 1.5
    # Exactly on a beat: every mode is a no-op.
    for mode in ("nearest", "previous", "next"):
        assert snap_to_beat(1.0, analysis, mode=mode) == 1.0
    # A midpoint resolves downward, deterministically.
    assert snap_to_beat(1.25, analysis, mode="nearest") == 1.0


def test_snap_clamps_outside_the_grid():
    """Times off either end of the grid clamp rather than run off it.

    Worth pinning because it is the behaviour a caller trips over when the
    music is shorter than the reel: every later cut clamps to the same final
    beat instead of raising.
    """
    analysis = _silent_analysis([1.0, 1.5, 2.0])

    assert snap_to_beat(0.1, analysis, mode="previous") == 1.0
    assert snap_to_beat(0.1, analysis, mode="nearest") == 1.0
    assert snap_to_beat(9.0, analysis, mode="next") == 2.0
    assert snap_to_beat(9.0, analysis, mode="nearest") == 2.0


def test_max_shift_refuses_a_runaway_snap_outside_the_grid():
    """The clamping above is unbounded, and that is what max_shift is for.

    Measured on a real track: a 141 s song whose first 14.8 s are an ambient
    intro has no beats in that stretch, so a cut at 10 s does not nudge onto
    the grid — it jumps 4.8 s forward onto the first beat, silently moving the
    edit somewhere nobody asked for.
    """
    analysis = _silent_analysis([14.768, 15.673, 16.625], bpm=66.0)
    beat = analysis.beat_interval

    assert snap_to_beat(10.0, analysis) == pytest.approx(14.768)
    assert snap_to_beat(10.0, analysis, max_shift=beat) == 10.0
    # Past the end clamps the same way, which is the music-shorter-than-the-reel case.
    assert snap_to_beat(60.0, analysis, max_shift=beat) == 60.0


def test_max_shift_still_allows_every_snap_inside_the_grid():
    """Inside the grid a nearest-snap is never more than half a beat away, so
    a one-beat range must not reject any genuine snap."""
    analysis = _silent_analysis([1.0, 1.5, 2.0, 2.5], bpm=120.0)
    beat = analysis.beat_interval

    for t, expected in ((1.1, 1.0), (1.4, 1.5), (1.74, 1.5), (2.3, 2.5)):
        assert snap_to_beat(t, analysis, max_shift=beat) == pytest.approx(expected)


def test_max_shift_leaves_the_far_end_of_a_segment_alone(tmp_path):
    """A segment starting before the music has one end on the grid and one
    that must keep its own time rather than being dragged."""
    analysis = _silent_analysis([14.768, 15.673, 16.625], bpm=66.0)

    (start, end), = snap_segments([(10.0, 15.6)], analysis,
                                  max_shift=analysis.beat_interval)

    assert start == 10.0
    assert end == pytest.approx(15.673)


def test_snap_segments_without_max_shift_is_unchanged():
    """The default must stay exactly what it was — max_shift is opt-in."""
    analysis = _silent_analysis([14.768, 15.673, 16.625], bpm=66.0)

    assert snap_segments([(10.0, 15.6)], analysis) == [
        pytest.approx((14.768, 15.673))]


def test_unknown_snap_mode_raises():
    """A typo'd mode must not silently mean `nearest`."""
    analysis = _silent_analysis([0.0, 0.5, 1.0])
    with pytest.raises(ValueError):
        snap_to_beat(0.3, analysis, mode="closest")
    with pytest.raises(ValueError):
        snap_segments([(0.0, 1.0)], analysis, mode="closest")


def test_snap_segments_never_returns_a_short_clip():
    """Snapping both ends can collapse a segment; min_duration must survive it.

    Two ends that round onto the same beat give a zero-length clip, which the
    cutter would emit as a frame or fail on. The guarantee is unconditional —
    including past the end of the grid, where there is no beat left to extend
    to and an off-grid end is still better than a clip too short to see.
    """
    beats = [i * 0.5 for i in range(9)]  # 0.0 .. 4.0
    analysis = _silent_analysis(beats)
    segments = [
        (0.05, 0.10),   # collapses onto beat 0
        (1.02, 1.20),   # collapses onto beat 1.0
        (1.60, 3.40),   # already long enough
        (3.90, 3.95),   # collapses at the very end of the grid
    ]

    snapped = snap_segments(segments, analysis, mode="nearest", min_duration=1.5)

    assert len(snapped) == len(segments)
    for start, end in snapped:
        assert end - start >= 1.5 - 1e-9
    # Everything reachable stays on the grid; only the run past the end is off it.
    grid = {round(b, 9) for b in beats}
    assert all(round(start, 9) in grid for start, _ in snapped)


def test_snap_segments_opens_a_collapsed_segment_without_a_minimum():
    """With no minimum, a collapsed segment still gets a full beat of length.

    Zero is a legal `min_duration` and must not be read as "a zero-length clip
    is acceptable"; the shortest musically meaningful span is one beat.
    """
    analysis = _silent_analysis([0.0, 0.5, 1.0, 1.5])
    (start, end), = snap_segments([(0.52, 0.55)], analysis)
    assert (start, end) == (0.5, 1.0)


def test_snap_segments_passes_through_without_a_grid():
    """No beats means no opinion about *placement* — but min_duration still holds.

    The two rules interact and the interaction is easy to get backwards: with
    nothing to snap to, times must come back untouched, yet a caller that asked
    for a minimum length asked for it unconditionally and would otherwise get a
    silently short clip on exactly the tracks where it cannot check.
    """
    analysis = _silent_analysis([], bpm=0.0)

    assert snap_segments([(1.0, 2.0), (3.0, 3.5)], analysis) == [(1.0, 2.0), (3.0, 3.5)]
    assert snap_segments([], analysis) == []

    stretched = snap_segments([(1.0, 2.0), (3.0, 3.5)], analysis, min_duration=1.5)
    assert stretched == [(1.0, 2.5), (3.0, 4.5)]


def test_beat_aligned_durations_spans_whole_bars():
    """Bars, not beats: the natural length for a cut that lands on a downbeat."""
    analysis = _silent_analysis([0.0, 0.5, 1.0], bpm=120.0, meter=4)
    assert beat_aligned_durations(analysis) == pytest.approx(2.0)
    assert beat_aligned_durations(analysis, bars=2) == pytest.approx(4.0)
    assert beat_aligned_durations(analysis, bars=0) == 0.0

    waltz = _silent_analysis([0.0, 0.5, 1.0], bpm=120.0, meter=3)
    assert beat_aligned_durations(waltz) == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def test_json_round_trip_preserves_the_grid(click_120, tmp_path):
    """A cached analysis must reload identically enough to cut against.

    The saved floats are rounded to shrink the envelope dump; the round trip is
    therefore exact to 1e-5, and this pins that so the rounding cannot quietly
    get coarser.
    """
    path, _ = click_120
    original = analyze_music(path, backend="numpy", log_fn=lambda *_: None)

    sidecar = save_analysis(original, tmp_path / "cache" / "analysis.json")
    restored = load_analysis(sidecar)

    assert restored.path == original.path
    assert restored.backend == original.backend
    assert restored.meter == original.meter
    assert restored.sample_rate == original.sample_rate
    assert restored.bpm == pytest.approx(original.bpm, abs=1e-5)
    assert restored.duration == pytest.approx(original.duration, abs=1e-5)
    assert restored.beat_interval == pytest.approx(original.beat_interval, abs=1e-8)
    assert restored.beats == pytest.approx(original.beats, abs=1e-5)
    assert restored.downbeats == pytest.approx(original.downbeats, abs=1e-5)
    assert restored.onset_envelope == pytest.approx(original.onset_envelope, abs=1e-4)
    assert restored.onset_times == pytest.approx(original.onset_times, abs=1e-5)
    assert len(restored.sections) == len(original.sections)
    for got, want in zip(restored.sections, original.sections):
        assert got.label == want.label
        assert got.start == pytest.approx(want.start, abs=1e-5)
        assert got.end == pytest.approx(want.end, abs=1e-5)
        assert got.energy == pytest.approx(want.energy, abs=1e-5)

    # A restored analysis is a working analysis, not just a data bag.
    assert snap_to_beat(3.3, restored) == pytest.approx(snap_to_beat(3.3, original))


def test_load_rejects_what_it_cannot_trust(tmp_path):
    """Half a beat grid is worse than none: bad sidecars must raise.

    A cutter given a truncated cache would place boundaries against a phantom
    tempo, and nothing downstream could tell that from a real one.
    """
    missing = tmp_path / "nope.json"
    with pytest.raises(ValueError):
        load_analysis(missing)

    garbage = tmp_path / "garbage.json"
    garbage.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        load_analysis(garbage)

    wrong_schema = tmp_path / "old.json"
    wrong_schema.write_text('{"schema": 0, "bpm": 120.0}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_analysis(wrong_schema)

    truncated = tmp_path / "truncated.json"
    truncated.write_text(
        '{"schema": %d, "bpm": 120.0, "beats": [0.0]}' % ma.SCHEMA_VERSION,
        encoding="utf-8")
    with pytest.raises(ValueError):
        load_analysis(truncated)


def test_saved_sections_survive_an_empty_analysis(tmp_path):
    """The degenerate analysis must serialise too — it is the one most likely
    to be cached, because silence is fast to analyse and cheap to re-read."""
    empty = MusicAnalysis(
        path="silent.wav", duration=0.0, sample_rate=SR, bpm=0.0, beats=[],
        downbeats=[], beat_interval=0.0, meter=4, onset_envelope=[],
        onset_times=[], sections=[Section(0.0, 1.0, 0.0, "low")], backend="numpy")
    restored = load_analysis(save_analysis(empty, tmp_path / "empty.json"))
    assert restored.bpm == 0.0
    assert restored.beats == []
    assert restored.sections[0].label == "low"


# ---------------------------------------------------------------------------
# The ffmpeg detour, for inputs the stdlib cannot read
# ---------------------------------------------------------------------------
def _ffmpeg_ok() -> bool:
    try:
        return subprocess.run([ffmpeg_exe(), "-version"],
                              capture_output=True, timeout=15).returncode == 0
    except Exception:
        return False


@pytest.mark.skipif(not _ffmpeg_ok(), reason="ffmpeg not available on this machine")
def test_non_pcm_input_decodes_through_ffmpeg(click_120, tmp_path):
    """A float wav is a wav the stdlib refuses; the analysis must still happen.

    `_read_pcm_wav` returning None is the hinge — if it ever raised instead,
    every compressed or float input would fail rather than fall back.
    """
    source, clicks = click_120
    converted = str(tmp_path / "float32.wav")
    subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-i", source,
                    "-c:a", "pcm_f32le", converted], check=True,
                   capture_output=True)

    assert ma._read_pcm_wav(converted) is None, "fixture is no longer non-PCM"

    logged = []
    analysis = analyze_music(converted, backend="numpy", log_fn=logged.append)
    assert abs(analysis.bpm - 120.0) < BPM_TOLERANCE
    assert _worst_alignment(analysis.beats, clicks) < BEAT_TOLERANCE
    assert any("ffmpeg" in line for line in logged)


@pytest.mark.skipif(not _ffmpeg_ok(), reason="ffmpeg not available on this machine")
def test_temp_wav_is_cleaned_up_after_decoding(tmp_path, monkeypatch):
    """The decoder writes a temp wav per call and must not leave it behind.

    Analysing a folder of tracks would otherwise fill the temp directory with
    one uncompressed copy of every file.
    """
    signal, _ = _click_track(120.0, 4.0)
    source = _write_wav(tmp_path / "src.wav", signal)
    compressed = str(tmp_path / "src.m4a")
    subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-i", source,
                    compressed], check=True, capture_output=True)

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr("tempfile.tempdir", str(scratch))

    analyze_music(compressed, backend="numpy", log_fn=lambda *_: None)
    assert list(scratch.iterdir()) == []


def test_no_ffmpeg_and_unreadable_input_raises(tmp_path, monkeypatch):
    """When there is no route to the samples, that is an error, not empty audio.

    Returning a zero-length signal here would look exactly like a silent track
    and the caller would cut a whole reel against an empty grid.
    """
    bogus = tmp_path / "not-audio.dat"
    bogus.write_bytes(b"this is not a media file")

    def _missing():
        raise FileNotFoundError("ffmpeg")

    monkeypatch.setattr(ma, "ffmpeg_exe", _missing)
    with pytest.raises((RuntimeError, FileNotFoundError)):
        analyze_music(str(bogus), backend="numpy", log_fn=lambda *_: None)


# ---------------------------------------------------------------------------
# Internals worth pinning directly
# ---------------------------------------------------------------------------
def test_onset_envelope_measures_attack_not_loudness():
    """A held chord must stay quiet in the envelope while faint clicks peak.

    This is the entire reason the envelope is spectral flux and not RMS, and it
    is the property that disappears the moment someone "simplifies" the stage
    to a loudness curve. The fixture is deliberately inverted — the sustained
    chord is some seventy times louder than the clicks — so a loudness-based
    envelope would rank them exactly backwards and every beat would land on
    the chord.
    """
    sample_rate = SR
    span = np.arange(3 * sample_rate)
    chord = np.zeros(3 * sample_rate)
    for frequency in (220.0, 277.0, 330.0, 440.0):
        chord += 0.22 * np.sin(2 * np.pi * frequency * span / sample_rate)
    clicks, _ = _click_track(120.0, 3.0, gain=0.15, seed=7)
    signal = np.concatenate([chord, clicks])

    loud = float(np.sqrt((chord ** 2).mean()))
    faint = float(np.sqrt((clicks ** 2).mean()))
    assert loud > 20 * faint, "fixture no longer inverts level against attack"

    envelope, times = ma._onset_envelope(signal.astype(np.float32), sample_rate)
    assert envelope.size == times.size

    sustain = envelope[(times > 0.5) & (times < 2.9)]
    clicky = envelope[times > 3.0]
    assert float(clicky.max()) > 0.5, "quiet attacks must still be peaks"
    assert float(sustain.max()) < 0.1 * float(clicky.max())


@pytest.mark.parametrize("bpm", [120.0, 140.0])
def test_tempo_estimate_resolves_finer_than_the_lag_grid(bpm):
    """The autocorrelation peak must be interpolated, not just argmax'd.

    Lags are whole frames, and at these tempi one frame is 2.5 to 3.5 BPM — so
    the integer peak alone cannot do better than that, and 140 BPM reads as
    143.6. The end-to-end tempo hides this, because refining the BPM from the
    tracked beats afterwards repairs it; the stage has to be checked on its own
    or the sub-frame fit can be deleted with every test still green.
    """
    signal, _ = _click_track(bpm, 24.0)
    envelope, _ = ma._onset_envelope(signal.astype(np.float32), SR)
    estimate, period = ma._estimate_tempo(envelope, SR)

    assert abs(estimate - bpm) < 0.5
    assert period == pytest.approx(60.0 * SR / (ma.HOP * estimate), rel=1e-6)
    assert period != round(period), "period landed exactly on a frame boundary"


def test_downsampling_rejects_content_above_the_new_nyquist():
    """Content that cannot survive the new rate must be filtered, not folded.

    Skipping the pre-filter is invisible on a click track and wrong on real
    music: a 16 kHz tone at 44.1 kHz reappears at 6 kHz after naive decimation,
    and the onset envelope reads that alias as an instrument. The check is
    two-sided — a 1 kHz tone has to come through essentially untouched, or the
    "filter" is just destroying the signal.
    """
    span = np.arange(44100)
    above = np.sin(2 * np.pi * 16000.0 * span / 44100.0).astype(np.float32)
    inside = np.sin(2 * np.pi * 1000.0 * span / 44100.0).astype(np.float32)

    def _rms(values):
        return float(np.sqrt((np.asarray(values, dtype=np.float64) ** 2).mean()))

    folded = ma._resample(above, 44100, ma.TARGET_SR)
    passed = ma._resample(inside, 44100, ma.TARGET_SR)

    assert folded.size == ma.TARGET_SR
    assert _rms(folded) < 0.5 * _rms(above), "out-of-band tone was not attenuated"
    assert _rms(passed) > 0.95 * _rms(inside), "in-band tone was damaged"


@pytest.mark.parametrize("samples", [1, 2, 3, 4, 17, 64])
@pytest.mark.parametrize("rate", [44100, 48000, 192000])
def test_resampling_never_returns_more_audio_than_it_was_given(samples, rate):
    """A file cannot get longer by being resampled down.

    `np.convolve(..., mode="same")` returns `max(len(y), taps)` elements, so
    anti-aliasing a signal shorter than its own kernel lengthens it: one sample
    at 44.1 kHz comes back as two and the file reports four times the audio it
    holds. The kernel is 4 taps at 44.1 kHz and 17 at 192 kHz, so how short is
    "short" moves with the input rate. Nothing else here reaches this path —
    the short-file test writes 22050 Hz fixtures, where the rates match and
    `_resample` returns on its first line.
    """
    out = ma._resample(np.ones(samples, dtype=np.float32), rate, ma.TARGET_SR)
    assert out.size <= math.ceil(samples * ma.TARGET_SR / rate)


def test_a_truncated_high_rate_file_reports_its_real_length(tmp_path):
    """The same bug end to end: a two-sample 44.1 kHz wav is not 4.5 ms long."""
    path = _write_wav(tmp_path / "truncated.wav", np.ones(2) * 0.5,
                      sample_rate=44100)
    analysis = analyze_music(path, backend="numpy", log_fn=lambda *_: None)

    assert analysis.duration <= 2.0 / 44100.0 + 1e-12
    assert analysis.bpm == 0.0
    assert analysis.beats == []


def test_no_beats_are_invented_over_silence(tmp_path):
    """The grid must stop where the music stops, at both ends.

    The dynamic program has to emit a beat every period all the way to the last
    frame, so without trimming a track with a silent intro and outro comes back
    with a full grid of beats over nothing — and the cutter dutifully places
    boundaries on them.
    """
    clicks, times = _click_track(120.0, 8.0)
    lead = np.zeros(4 * SR)
    signal = np.concatenate([lead, clicks, np.zeros(4 * SR)])
    path = _write_wav(tmp_path / "gapped.wav", signal)

    analysis = analyze_music(path, backend="numpy", log_fn=lambda *_: None)
    first_click = 4.0 + times[0]
    last_click = 4.0 + times[-1]

    assert abs(analysis.bpm - 120.0) < BPM_TOLERANCE
    assert analysis.beats[0] > first_click - 0.5
    assert analysis.beats[-1] < last_click + 0.5
    # 16 s of file, 8 s of music: an untrimmed grid would be about twice as long.
    assert len(analysis.beats) <= len(times) + 2


def test_frame_times_match_the_hop_grid(click_120):
    """onset_times must be the time axis of onset_envelope, hop by hop.

    They are returned as separate lists; if they ever disagree in length or
    spacing, every plot and every downstream index lines up against the wrong
    instant.
    """
    path, _ = click_120
    analysis = analyze_music(path, backend="numpy", log_fn=lambda *_: None)

    assert len(analysis.onset_times) == len(analysis.onset_envelope)
    step = ma.HOP / float(analysis.sample_rate)
    assert np.allclose(np.diff(analysis.onset_times), step)
    assert analysis.onset_times[0] == 0.0
    assert max(analysis.onset_envelope) == pytest.approx(1.0)
    expected = math.ceil(analysis.duration / step)
    assert abs(len(analysis.onset_times) - expected) <= 2
