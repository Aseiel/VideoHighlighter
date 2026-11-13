"""Subtitle cues timed to shot changes (modules/audio/shot_snap.py)."""
from modules.audio import shot_snap as ss
from modules.audio import transcript_srt

FPS = 25.0
F = 1 / FPS


def _cue(start, end, text="line"):
    return {"start": start, "end": end, "text": text}


def _times(cues):
    return [(round(c["start"] * FPS), round(c["end"] * FPS)) for c in cues]


def test_a_start_just_after_a_cut_moves_back_onto_it():
    out, n = ss.snap([_cue(10.2, 12.0)], [10.0], FPS)          # 5 frames late
    assert _times(out) == [(250, 300)] and n["starts"] == 1


def test_a_start_just_before_a_cut_moves_forward_onto_it():
    out, _ = ss.snap([_cue(9.8, 12.0)], [10.0], FPS)
    assert _times(out) == [(250, 300)]


def test_a_start_far_from_any_cut_stays():
    out, n = ss.snap([_cue(11.0, 13.0)], [10.0], FPS)          # 25 frames off
    assert _times(out) == [(275, 325)] and n == {"starts": 0, "ends": 0, "gaps": 0}


def test_an_end_just_past_a_cut_ends_two_frames_before_it():
    out, n = ss.snap([_cue(8.0, 10.2)], [10.0], FPS)
    assert _times(out) == [(200, 248)] and n["ends"] == 1


def test_an_end_just_before_a_cut_is_held_to_two_frames_before_it():
    out, _ = ss.snap([_cue(8.0, 9.7)], [10.0], FPS)
    assert _times(out) == [(200, 248)]


def test_a_short_line_is_held_past_the_cut_rather_than_cut_shorter():
    out, _ = ss.snap([_cue(9.4, 10.2)], [10.0], FPS)           # 0.8 s; pulled back = 0.52 s
    assert _times(out) == [(235, 250 + ss.SNAP_FRAMES)]


def test_the_line_before_gives_way_when_the_next_starts_on_the_cut():
    # A line running 7 frames into the next shot, and the next line starting
    # 9 frames after the cut: both change with the picture.
    out, _ = ss.snap([_cue(36.58, 39.74), _cue(39.81, 42.0)], [39.51], FPS)
    cut = round(39.51 * FPS)
    assert _times(out) == [(round(36.58 * FPS), cut - ss.GAP_FRAMES), (cut, 1050)]


def test_a_small_gap_is_closed_to_two_frames_but_not_across_a_cut():
    out, _ = ss.snap([_cue(1.0, 2.0), _cue(2.2, 3.0)], [], FPS)
    assert _times(out) == [(25, 53), (55, 75)]
    out, _ = ss.snap([_cue(1.0, 2.0), _cue(2.4, 3.6)], [2.3], FPS)
    assert out[0]["end"] * FPS <= round(2.3 * FPS) - ss.GAP_FRAMES + 1e-6


def test_cues_never_overlap_and_stay_in_order():
    cues = [_cue(t, t + 1.1) for t in (0.0, 1.0, 2.05, 3.3, 4.1)]
    out, _ = ss.snap(cues, [1.05, 2.1, 3.25, 4.0], FPS)
    for a, b in zip(out, out[1:]):
        assert a["start"] < a["end"] <= b["start"] - ss.GAP_FRAMES * F + 1e-9


def test_the_srt_writer_snaps_only_when_given_shot_changes():
    segs = [{"start": 10.2, "end": 12.0, "text": "Hello there, friend."}]
    plain = transcript_srt.create_srt_content(segs)
    snapped = transcript_srt.create_srt_content(segs, shot_changes=([10.0], FPS),
                                                log_fn=lambda *_: None)
    assert "00:00:10,000 -->" not in plain
    assert "00:00:10,000 -->" in snapped


# --- "Make subtitles" with the setting on (analysis_ondemand.run_subtitles) --

def _run_subtitles(monkeypatch, tmp_path, **kwargs):
    from modules.report import analysis_ondemand as aod
    from modules.audio import shot_snap
    seen = {"found": 0, "srt": []}
    segs = [{"start": 0.0, "end": 2.0, "text": "first line"}]
    monkeypatch.setattr(aod, "cached_transcript",
                        lambda *a, **k: {"segments": segs, "language": "en"})
    monkeypatch.setattr(aod, "_write_transcript_sidecar", lambda *a, **k: None)

    def found(*a, **k):
        seen["found"] += 1
        return ([1.0], FPS)

    monkeypatch.setattr(shot_snap, "shot_changes", found)
    monkeypatch.setattr(transcript_srt, "create_srt_file",
                        lambda segments, path, **kw: seen["srt"].append(kw))
    aod.run_subtitles(str(tmp_path / "movie.mp4"), language="en",
                      log=lambda *_: None, **kwargs)
    return seen


def test_make_subtitles_finds_shot_changes_only_when_asked(monkeypatch, tmp_path):
    off = _run_subtitles(monkeypatch, tmp_path)
    assert off["found"] == 0 and off["srt"][0]["shot_changes"] is None
    on = _run_subtitles(monkeypatch, tmp_path, snap_to_shots=True)
    assert on["found"] == 1 and on["srt"][0]["shot_changes"] == ([1.0], FPS)
