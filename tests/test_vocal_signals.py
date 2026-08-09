"""
Tests for `modules.vocal_signals` — per-second vocal brightness and onset.

Driven by synthetic audio rather than a fixture video, so the suite keeps its
"no heavy deps, seconds not minutes" property and so each property is tested
against a signal whose answer is known by construction rather than by having
been listened to.

What is pinned here is the *contract*, not the numbers: that brightness tracks
spectral centre of mass, that onset tracks a rise in level, that both are
normalised against the video's own voice rather than an absolute, and that the
gate rejects broadband noise. The thresholds themselves were fitted on labelled
material and are recorded in the module, where their justification is.
"""

from __future__ import annotations

import numpy as np
import pytest

from modules import vocal_signals as vs


SR = vs.SAMPLE_RATE


def _tone(freq, seconds, amplitude=0.3, sr=SR):
    """A harmonic-rich periodic tone — voice-like enough to pass the gate."""
    t = np.arange(int(seconds * sr)) / sr
    wave = np.zeros_like(t)
    for harmonic, weight in enumerate((1.0, 0.5, 0.25, 0.12), start=1):
        wave += weight * np.sin(2 * np.pi * freq * harmonic * t)
    return amplitude * wave / np.abs(wave).max()


def _noise(seconds, amplitude=0.3, sr=SR, seed=0):
    return amplitude * np.random.default_rng(seed).normal(size=int(seconds * sr))


def _write(tmp_path, samples, name="a.wav"):
    import wave

    path = tmp_path / name
    data = np.clip(samples, -1.0, 1.0)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes((data * 32767).astype(np.int16).tobytes())
    return str(path)


# ----------------------------------------------------------------- features

def test_brightness_tracks_spectral_centre(tmp_path):
    """A higher tone must measure brighter. That is the whole of the claim."""
    path = _write(tmp_path, np.concatenate([_tone(200, 3), _tone(900, 3)]))
    f = vs.per_second_features(path)
    low, high = f["brightness"][:3].mean(), f["brightness"][3:6].mean()
    assert high > low * 1.5, f"900 Hz measured {high:.0f} Hz, 200 Hz {low:.0f} Hz"


def test_periodicity_separates_tone_from_noise(tmp_path):
    path = _write(tmp_path, np.concatenate([_tone(300, 3), _noise(3)]))
    f = vs.per_second_features(path)
    assert f["periodicity"][:3].mean() > 0.5
    assert f["periodicity"][3:6].mean() < 0.3


def test_gate_rejects_broadband_noise_and_keeps_a_tone(tmp_path):
    """The gate's actual job: drop the impacts, keep the voice."""
    path = _write(tmp_path, np.concatenate([_tone(300, 3), _noise(3)]))
    vocal = vs.vocal_mask(vs.per_second_features(path))
    assert vocal[:3].all(), "a harmonic tone was rejected as non-vocal"
    assert not vocal[3:6].any(), "broadband noise passed the vocal gate"


# -------------------------------------------------------------------- onset

def test_onset_measures_a_rise_not_a_level():
    """A step up scores; staying loud does not."""
    level = np.array([-30.0, -30.0, -10.0, -10.0, -10.0])
    onset = vs.onset_curve(level)
    assert onset[2] == pytest.approx(20.0)
    # Still loud at index 4, but it rose two seconds ago, so nothing is rising.
    assert onset[4] == pytest.approx(0.0)


def test_onset_sees_a_rise_spread_over_two_seconds():
    """Differencing against the previous second alone would score this twice
    as two small rises; against the window minimum it is one large one."""
    level = np.array([-30.0, -20.0, -10.0])
    assert vs.onset_curve(level)[2] == pytest.approx(20.0)


def test_onset_never_looks_before_the_start():
    assert vs.onset_curve(np.array([-10.0]))[0] == 0.0


# ------------------------------------------------------------ normalisation

def test_robust_z_is_relative_to_the_reference():
    """The same absolute value scores differently against different voices —
    which is the entire reason the score is not in hertz."""
    quiet = np.array([100.0, 110.0, 120.0, 130.0])
    loud = np.array([500.0, 510.0, 520.0, 530.0])
    assert vs.robust_z(np.array([520.0]), quiet)[0] > 10
    assert abs(vs.robust_z(np.array([520.0]), loud)[0]) < 2


def test_robust_z_survives_a_constant_reference():
    """No variation must not divide out to an infinity.

    Nor to a number that is finite and absurd, which is what a bare epsilon
    floor produced: on a reference with no spread, a tenth of a unit of drift
    came back as a z-score in the millions and any threshold a user set was
    meaningless. The floor is in the units of the quantity for that reason.
    """
    z = vs.robust_z(np.array([5.0]), np.array([1.0, 1.0, 1.0]), 0.5)
    assert np.isfinite(z[0])
    assert abs(z[0]) < 20, f"constant reference produced z={z[0]}"


def test_robust_z_is_not_dragged_by_the_values_it_is_measuring():
    """Median and MAD, so a handful of extremes cannot move their own centre."""
    ordinary = np.full(50, 100.0) + np.arange(50) * 0.1
    with_extremes = np.concatenate([ordinary, np.full(5, 10000.0)])
    z_clean = vs.robust_z(np.array([120.0]), ordinary)[0]
    z_polluted = vs.robust_z(np.array([120.0]), with_extremes)[0]
    assert z_polluted == pytest.approx(z_clean, rel=0.25)


# --------------------------------------------------------------- end to end

def test_analyse_reports_zero_effort_outside_the_gate(tmp_path, monkeypatch):
    """A score describing something that is not a voice is not reported."""
    samples = np.concatenate([_tone(300, 20), _noise(20), _tone(700, 20)])
    path = _write(tmp_path, samples)
    monkeypatch.setattr(vs, "extract_audio",
                        lambda video, wav: __import__("shutil").copyfile(path, wav))
    monkeypatch.setattr(vs, "MIN_VOCAL_SECONDS", 5)

    result = vs.analyse("ignored.mp4")
    effort = np.array(result["curves"]["effort"])
    vocal = np.array(result["curves"]["vocal"], dtype=bool)

    assert not vocal[20:40].any(), "noise passed the gate"
    assert (effort[~vocal] == 0).all(), "a gated-out second carried a score"
    # The brighter of the two tones must score above the duller one.
    assert effort[40:60].mean() > effort[:20].mean()


def test_analyse_declines_to_normalise_on_too_little_voice(tmp_path, monkeypatch):
    """Below the floor the z-scores would describe a handful of samples."""
    path = _write(tmp_path, np.concatenate([_tone(300, 3), _noise(30)]))
    monkeypatch.setattr(vs, "extract_audio",
                        lambda video, wav: __import__("shutil").copyfile(path, wav))

    result = vs.analyse("ignored.mp4")
    assert result["enough_voice"] is False
    assert all(v == 0 for v in result["curves"]["effort"])


def test_top_moments_skips_gated_seconds_and_spaces_them(tmp_path):
    result = {"curves": {
        "effort": [0.0, 5.0, 4.9, 4.8, 0.0, 4.0, 0.0, 9.0],
        "vocal": [True] * 7 + [False],
        "brightness_z": [0.0] * 8,
        "onset_z": [0.0] * 8,
    }}
    picked = vs.top_moments(result, count=5, spacing=3)
    seconds = [p["second"] for p in picked]
    assert 7 not in seconds, "a gated-out second was surfaced"
    assert seconds == [1, 5], "moments were not spaced apart"


# ------------------------------------------------------------- density

def test_density_prefers_clustered_loudness_over_an_isolated_peak():
    """The measurement the waveform was showing: many loud seconds together.

    A single very loud second and a sustained loud stretch can carry the same
    peak, and peak is what earlier attempts here ranked on. These must not
    score the same.
    """
    quiet = np.full(60, -30.0)
    spike = quiet.copy()
    spike[30] = -5.0
    cluster = quiet.copy()
    cluster[25:36] = -12.0

    s = vs.density_curve(spike)[30]
    c = vs.density_curve(cluster)[30]
    assert c > s, f"clustered {c:.3f} did not beat isolated spike {s:.3f}"


def test_density_needs_both_halves():
    """Continuous murmur is dense but not loud; it must not score like a burst."""
    murmur = np.full(60, -30.0)
    murmur[::2] = -29.5                       # dense, barely above its own median
    burst = np.full(60, -30.0)
    burst[25:36] = -10.0
    assert vs.density_curve(burst)[30] > vs.density_curve(murmur)[30]


def test_percentile_rank_is_file_relative():
    """So `min: 90` means the same thing on two differently mastered files."""
    quiet = vs.percentile_rank(np.array([1.0, 2.0, 3.0, 4.0, 100.0]))
    loud = vs.percentile_rank(np.array([500.0, 600.0, 700.0, 800.0, 9000.0]))
    assert quiet.tolist() == loud.tolist()
    assert quiet[-1] == 100.0 and quiet[0] == 0.0


def test_percentile_rank_handles_degenerate_input():
    assert vs.percentile_rank(np.zeros(0)).tolist() == []
    assert np.isfinite(vs.percentile_rank(np.ones(5))).all()


# -------------------------------------------------------- waveform peaks

def test_waveform_peaks_finds_local_maxima_above_the_cut():
    """Three bursts, well separated, must come back as three peaks."""
    fine = np.full(300, 0.01)          # 30 s at 10 Hz
    for centre in (50, 150, 250):
        fine[centre] = 1.0
    peaks = vs.waveform_peaks(fine)
    assert [round(p, 1) for p in peaks] == [5.0, 15.0, 25.0]


def test_waveform_peaks_suppresses_neighbours_within_the_gap():
    """One loud region is one stop, not a cluster of them — the same
    non-maximum suppression the timeline arrows use."""
    fine = np.full(300, 0.01)
    fine[100] = 1.0
    fine[104] = 0.9                     # 0.4 s later, inside the 1 s gap
    fine[140] = 0.95                    # 4 s later, outside it
    peaks = vs.waveform_peaks(fine)
    assert len(peaks) == 2, f"expected 2 distinct peaks, got {peaks}"


def test_waveform_peaks_are_relative_to_the_file():
    """A quiet recording and a loud one with the same shape give the same peaks.

    This is the property that lets one sensitivity work on every file, and the
    reason the normalisation is a percentile range rather than an absolute.
    """
    shape = np.full(300, 0.01)
    for c in (50, 150, 250):
        shape[c] = 1.0
    assert vs.waveform_peaks(shape) == vs.waveform_peaks(shape * 0.01)


def test_peak_density_counts_within_the_window():
    peaks = [1.0, 2.0, 3.0, 50.0]
    d = vs.peak_density_curve(peaks, seconds=60, window=20)
    assert d[2] == 3.0, "the three clustered peaks were not counted together"
    assert d[50] == 1.0, "the isolated peak was counted with the cluster"


def test_peak_density_handles_empty_input():
    assert vs.peak_density_curve([], seconds=10).tolist() == [0.0] * 10
    assert vs.waveform_peaks(np.zeros(0)) == []


def test_curves_never_outrun_the_audio_they_describe():
    """A clip shorter than the analysis window must not produce a longer curve.

    `np.convolve(..., "same")` returns max(len(signal), len(kernel)), so a 5 s
    clip and a 20 s window silently yielded 20 values — a curve misaligned
    against every other per-second signal a rule tests alongside it.
    """
    short = np.full(5, -20.0)
    assert len(vs.density_curve(short, window=10)) == 5
    assert len(vs.peak_density_curve([1.0], seconds=5, window=20)) == 5
