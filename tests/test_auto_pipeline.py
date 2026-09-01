"""
Tests for modules.auto_pipeline — the card-to-film orchestrator.

The engine is injected as a stub (``highlight_runner``) throughout, because
nothing here is a test of detection quality: what is being tested is the
orchestration contract — that stages run in order, that a resumed run skips
exactly the work that is still on disk and redoes the work that is not, that
cancellation leaves recoverable state, and that an optional stage failing still
hands back the film that was already made.

That last one is the behaviour most worth protecting. The expensive part of
this pipeline is detection; a music filter failing at the very end must never
be allowed to discard it.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess

import pytest

from modules.app_paths import ffmpeg_exe
from modules.auto_pipeline import (
    DONE,
    FAILED,
    SKIPPED,
    STAGE_COMBINE,
    STAGE_HIGHLIGHT,
    STAGE_INGEST,
    STAGE_MUSIC,
    STAGE_MUSIC_MIX,
    JobState,
    PipelineCancelled,
    Stage,
    apply_script_to_config,
    job_path,
    load_job,
    run_auto_pipeline,
    save_job,
)

FFMPEG = ffmpeg_exe()


def _ffmpeg_ok() -> bool:
    try:
        return subprocess.run([FFMPEG, "-version"],
                              capture_output=True).returncode == 0
    except OSError:
        return False


# The combine stage runs the real modules.combine_videos, which runs real
# ffmpeg. Feeding it fake bytes would only ever test that ffmpeg rejects
# garbage, so the stub engine emits genuine (tiny) clips instead and these
# tests skip outright where ffmpeg is unavailable.
pytestmark = pytest.mark.skipif(not _ffmpeg_ok(), reason="ffmpeg not available")


@pytest.fixture(scope="module")
def tiny_clip(tmp_path_factory):
    """One real 0.3 s 64x64 clip with silent audio, made once and copied.

    Real enough for the combiner to normalize and concat; small enough that
    making it costs a fraction of a second.
    """
    path = tmp_path_factory.mktemp("fixture") / "tiny.mp4"
    subprocess.run(
        [FFMPEG, "-y", "-v", "error",
         "-f", "lavfi", "-i", "testsrc=size=64x64:rate=10:duration=0.3",
         "-f", "lavfi", "-i", "anullsrc=channel_layout=mono:sample_rate=48000",
         "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-c:a", "aac", str(path)],
        check=True, capture_output=True)
    return str(path)


def _sources(tmp_path, n=2):
    """n placeholder source files. Content is irrelevant — the stub engine
    never decodes them."""
    src = tmp_path / "src"
    src.mkdir(exist_ok=True)
    paths = []
    for i in range(n):
        p = src / f"GX01{3762 + i}.MP4"
        p.write_bytes(b"video" * 64)
        paths.append(str(p))
    return paths


def _stub_engine(out_dir, clip, *, calls=None, fail=False):
    """A stand-in for pipeline.run_highlighter that emits one real clip per
    input and records that it was called."""
    def runner(paths, gui_config, log_fn, progress_fn, cancel_flag):
        if calls is not None:
            calls.append(list(paths))
        if fail:
            raise RuntimeError("detector exploded")
        os.makedirs(out_dir, exist_ok=True)
        made = []
        for p in paths:
            out = os.path.join(out_dir, os.path.basename(p).replace(".MP4", "_hl.mp4"))
            shutil.copy2(clip, out)
            made.append((p, out))
        return made
    return runner


def _run(tmp_path, tiny_clip, **kwargs):
    dest = kwargs.pop("dest", None) or str(tmp_path / "out")
    defaults = dict(
        dest_root=dest,
        source_paths=_sources(tmp_path),
        highlight_runner=_stub_engine(str(tmp_path / "hl"), tiny_clip),
        log_fn=lambda *_: None,
    )
    defaults.update(kwargs)
    return run_auto_pipeline(**defaults)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_runs_end_to_end_and_produces_a_film(tmp_path, tiny_clip):
    result = _run(tmp_path, tiny_clip)

    assert result.ok
    assert os.path.exists(result.output)
    assert result.state.stages[STAGE_HIGHLIGHT].status == DONE
    assert result.state.stages[STAGE_COMBINE].status == DONE


def test_a_single_highlight_is_copied_not_re_encoded(tmp_path, tiny_clip):
    """Sending one clip through the combiner would cost a generation of
    quality for no benefit."""
    result = _run(tmp_path, tiny_clip, source_paths=_sources(tmp_path, n=1))

    assert result.ok
    assert os.path.basename(result.output) == "film.mp4"
    # Byte-identical to the engine's clip proves it was copied, not re-encoded.
    assert open(result.output, "rb").read() == open(tiny_clip, "rb").read()


def test_job_state_is_written_next_to_the_output(tmp_path, tiny_clip):
    result = _run(tmp_path, tiny_clip)

    path = job_path(result.state.root)
    assert os.path.exists(path)
    data = json.load(open(path, encoding="utf-8"))
    assert data["version"] == 1
    assert data["stages"][STAGE_HIGHLIGHT]["status"] == DONE


def test_source_paths_mode_skips_the_ingest_stage(tmp_path, tiny_clip):
    result = _run(tmp_path, tiny_clip)

    assert result.state.stages[STAGE_INGEST].status == SKIPPED


def test_requires_a_source(tmp_path):
    with pytest.raises(ValueError, match="card= or source_paths="):
        run_auto_pipeline(dest_root=str(tmp_path / "out"), log_fn=lambda *_: None)


# ---------------------------------------------------------------------------
# Resume — the reason the module exists
# ---------------------------------------------------------------------------

def test_resume_skips_work_already_on_disk(tmp_path, tiny_clip):
    dest = str(tmp_path / "out")
    calls: list[list[str]] = []
    runner = _stub_engine(str(tmp_path / "hl"), tiny_clip, calls=calls)

    _run(tmp_path, tiny_clip, dest=dest, highlight_runner=runner)
    assert len(calls) == 1

    _run(tmp_path, tiny_clip, dest=dest, highlight_runner=runner)

    assert len(calls) == 1, "second run re-ran detection instead of resuming"


def test_resume_redoes_a_stage_whose_output_was_deleted(tmp_path, tiny_clip):
    """A stage recorded as done but whose file is gone must be redone — not
    marched past into a confusing missing-file error downstream."""
    dest = str(tmp_path / "out")
    calls: list[list[str]] = []
    runner = _stub_engine(str(tmp_path / "hl"), tiny_clip, calls=calls)

    first = _run(tmp_path, tiny_clip, dest=dest, highlight_runner=runner)
    for p in first.state.highlights:
        os.remove(p)

    _run(tmp_path, tiny_clip, dest=dest, highlight_runner=runner)

    assert len(calls) == 2


def test_resume_false_redoes_everything(tmp_path, tiny_clip):
    dest = str(tmp_path / "out")
    calls: list[list[str]] = []
    runner = _stub_engine(str(tmp_path / "hl"), tiny_clip, calls=calls)

    _run(tmp_path, tiny_clip, dest=dest, highlight_runner=runner)
    _run(tmp_path, tiny_clip, dest=dest, highlight_runner=runner, resume=False)

    assert len(calls) == 2


def test_unreadable_job_state_starts_fresh_instead_of_failing(tmp_path, tiny_clip):
    dest = tmp_path / "out"
    dest.mkdir()
    (dest / "job.json").write_text("{ not json", encoding="utf-8")
    logged: list[str] = []

    result = _run(tmp_path, tiny_clip, dest=str(dest), log_fn=logged.append)

    assert result.ok
    assert any("Ignoring unusable job state" in m for m in logged)


def test_future_state_version_is_rejected(tmp_path):
    path = tmp_path / "job.json"
    path.write_text(json.dumps({"version": 99, "job_id": "x", "root": "y"}),
                    encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported job state version"):
        load_job(str(path))


def test_state_survives_a_save_load_round_trip(tmp_path):
    state = JobState(job_id="j1", root=str(tmp_path))
    state.stage(STAGE_HIGHLIGHT).status = DONE
    state.stage(STAGE_HIGHLIGHT).outputs = ["a.mp4"]
    state.clips = ["c1.MP4"]

    save_job(state)
    loaded = load_job(job_path(str(tmp_path)))

    assert loaded.job_id == "j1"
    assert loaded.clips == ["c1.MP4"]
    assert loaded.stages[STAGE_HIGHLIGHT].status == DONE
    assert loaded.stages[STAGE_HIGHLIGHT].outputs == ["a.mp4"]


def test_a_resumed_run_reloads_the_beat_grid(tmp_path, tiny_clip, monkeypatch):
    """A skipped stage returns nothing, so the analysis has to come back off
    disk — otherwise quantising is silently dropped on exactly the runs most
    likely to want it, the second and later attempts at the same film."""
    import sys
    import types

    dest = str(tmp_path / "out")
    music = tmp_path / "track.mp3"
    music.write_bytes(b"ID3fake")
    seen: list[str] = []

    class _Fake:
        bpm, beat_interval, meter = 120.0, 0.5, 4
        beats = [0.0, 0.5, 1.0, 1.5]
        downbeats = [0.0]

    fake = types.ModuleType("modules.music_analysis")
    fake.analyze_music = lambda *a, **k: _Fake()
    fake.save_analysis = lambda _a, path: (open(path, "w").write("{}"), path)[1]
    fake.load_analysis = lambda path: (seen.append(path), _Fake())[1]
    monkeypatch.setitem(sys.modules, "modules.music_analysis", fake)

    runner = _stub_engine(str(tmp_path / "hl"), tiny_clip)
    _run(tmp_path, tiny_clip, dest=dest, highlight_runner=runner,
         music_path=str(music))
    assert seen == [], "first run analysed rather than loaded"

    logged: list[str] = []
    _run(tmp_path, tiny_clip, dest=dest, highlight_runner=runner,
         music_path=str(music), log_fn=logged.append)

    assert seen, "the saved beat grid was never reloaded"
    assert any("Reusing the beat grid" in m for m in logged)


def test_changing_the_render_settings_rebuilds_the_reel(tmp_path, tiny_clip):
    """A resume must not hand back the previous render just because a file of
    the right name exists. Asking for crossfades and getting the hard-cut reel
    from last time is only discovered after watching it."""
    dest = str(tmp_path / "out")
    calls: list[list[str]] = []
    runner = _stub_engine(str(tmp_path / "hl"), tiny_clip, calls=calls)

    first = _run(tmp_path, tiny_clip, dest=dest, highlight_runner=runner)
    before = os.path.getmtime(first.output)

    second = _run(tmp_path, tiny_clip, dest=dest, highlight_runner=runner,
                  transition="crossfade", transition_duration=0.1)

    assert len(calls) == 1, "detection was needlessly redone"
    assert second.state.stages[STAGE_COMBINE].status == DONE
    assert os.path.getmtime(second.output) > before, "the reel was not rebuilt"


def test_an_unchanged_request_still_reuses_the_reel(tmp_path, tiny_clip):
    """The control for the test above — fingerprinting must not defeat resume
    for a run that asked for exactly the same thing."""
    dest = str(tmp_path / "out")
    runner = _stub_engine(str(tmp_path / "hl"), tiny_clip)

    first = _run(tmp_path, tiny_clip, dest=dest, highlight_runner=runner)
    before = os.path.getmtime(first.output)

    second = _run(tmp_path, tiny_clip, dest=dest, highlight_runner=runner)

    assert os.path.getmtime(second.output) == before


def test_stage_satisfaction_requires_the_files_to_exist(tmp_path):
    present = tmp_path / "here.mp4"
    present.write_bytes(b"x")

    assert Stage(name="s", status=DONE, outputs=[str(present)]).is_satisfied
    assert not Stage(name="s", status=DONE, outputs=[str(tmp_path / "gone.mp4")]).is_satisfied
    assert not Stage(name="s", status="pending", outputs=[str(present)]).is_satisfied


# ---------------------------------------------------------------------------
# Failure and cancellation
# ---------------------------------------------------------------------------

def test_a_failing_required_stage_stops_the_run(tmp_path, tiny_clip):
    with pytest.raises(RuntimeError, match="detector exploded"):
        _run(tmp_path, tiny_clip, highlight_runner=_stub_engine(str(tmp_path / "hl"), tiny_clip, fail=True))


def test_a_failed_stage_is_recorded_in_the_state_file(tmp_path, tiny_clip):
    dest = str(tmp_path / "out")
    with pytest.raises(RuntimeError):
        _run(tmp_path, tiny_clip, dest=dest, highlight_runner=_stub_engine(str(tmp_path / "hl"), tiny_clip, fail=True))

    state = load_job(job_path(dest))
    assert state.stages[STAGE_HIGHLIGHT].status == FAILED
    assert "detector exploded" in state.stages[STAGE_HIGHLIGHT].error


def test_engine_returning_nothing_is_an_error_not_an_empty_film(tmp_path, tiny_clip):
    """Silently producing a zero-clip reel would look like success."""
    def empty(paths, gui_config, log_fn, progress_fn, cancel_flag):
        return []

    with pytest.raises(RuntimeError, match="no highlight clips"):
        _run(tmp_path, tiny_clip, highlight_runner=empty)


def test_cancel_raises_and_leaves_resumable_state(tmp_path, tiny_clip):
    dest = str(tmp_path / "out")

    with pytest.raises(PipelineCancelled):
        _run(tmp_path, tiny_clip, dest=dest, cancel_check=lambda: True)

    state = load_job(job_path(dest))
    assert state.stages[STAGE_HIGHLIGHT].status != DONE


def test_music_failure_still_returns_the_silent_reel(tmp_path, tiny_clip, monkeypatch):
    """The expensive work is detection. Losing the film to an audio filter
    error at the last step would be absurd.

    The music is laid inside the combine stage now (that stage is already
    re-encoding, so a separate pass would rewrite the whole reel again), which
    means this property lives in ``build_reel``'s ``music_optional`` rather than
    in a stage that is allowed to fail. The property is the same either way:
    the reel survives.
    """
    import sys
    import types

    music = tmp_path / "track.mp3"
    music.write_bytes(b"ID3fake")

    # Both music paths are stubbed to fail: the mux is the one under test, and
    # the analysis is stubbed out so the test does not depend on a real decoder
    # being able to read seven bytes of fake mp3.
    def boom(*_args, **_kwargs):
        raise RuntimeError("ffmpeg music mux failed")

    bad_music = types.ModuleType("modules.music_track")
    bad_music.apply_music = boom
    monkeypatch.setitem(sys.modules, "modules.music_track", bad_music)

    bad_analysis = types.ModuleType("modules.music_analysis")
    bad_analysis.analyze_music = boom
    bad_analysis.save_analysis = boom
    monkeypatch.setitem(sys.modules, "modules.music_analysis", bad_analysis)

    result = _run(tmp_path, tiny_clip, music_path=str(music))

    # The analysis stage failing is what makes the run not-ok; the combine
    # stage absorbs the mux failure and still produces the reel.
    assert not result.ok
    assert result.state.stages[STAGE_MUSIC].status == FAILED
    assert result.state.stages[STAGE_COMBINE].status == DONE
    assert os.path.exists(result.output), "the silent reel was thrown away"
    assert result.output == result.state.reel


def test_a_broken_script_does_not_strand_the_footage(tmp_path, tiny_clip, monkeypatch):
    script = tmp_path / "script.yaml"
    script.write_text("nonsense: [", encoding="utf-8")
    logged: list[str] = []

    result = _run(tmp_path, tiny_clip, script_path=str(script), log_fn=logged.append)

    assert result.ok
    assert os.path.exists(result.output)
    assert any("Script ignored" in m for m in logged)


# ---------------------------------------------------------------------------
# Progress reporting
# ---------------------------------------------------------------------------

def test_progress_is_monotonic_and_ends_at_one(tmp_path, tiny_clip):
    seen: list[float] = []

    _run(tmp_path, tiny_clip, progress_fn=lambda f, d: seen.append(f))

    assert seen
    assert seen[-1] == 1.0
    assert all(0.0 <= f <= 1.0 for f in seen)
    assert seen == sorted(seen), "progress went backwards"


def test_stage_callback_reports_every_transition(tmp_path, tiny_clip):
    events: list[tuple[str, str]] = []

    _run(tmp_path, tiny_clip, stage_fn=lambda name, status, detail: events.append((name, status)))

    names = [n for n, _ in events]
    assert STAGE_HIGHLIGHT in names
    assert STAGE_COMBINE in names
    assert ("highlight", DONE) in events


# ---------------------------------------------------------------------------
# Script -> config translation
# ---------------------------------------------------------------------------

class _FakeBeat:
    def __init__(self, objects=(), actions=(), keywords=()):
        self.objects, self.actions, self.keywords = list(objects), list(actions), list(keywords)


class _FakeScript:
    def __init__(self, beats, target=0.0):
        self.beats = beats
        self.target_duration = target


def test_script_match_terms_merge_into_the_config(tmp_path):
    script = _FakeScript([_FakeBeat(objects=["boat"], keywords=["start"]),
                          _FakeBeat(objects=["water"], actions=["running"])])

    merged = apply_script_to_config(script, {})

    assert merged["objects"]["interesting"] == ["boat", "water"]
    assert merged["actions"]["interesting"] == ["running"]
    assert merged["keywords"]["interesting"] == ["start"]


def test_script_terms_are_deduplicated_and_appended_to_existing(tmp_path):
    script = _FakeScript([_FakeBeat(objects=["boat"]), _FakeBeat(objects=["boat"])])
    config = {"objects": {"interesting": ["boat", "dog"], "confidence": 30}}

    merged = apply_script_to_config(script, config)

    assert merged["objects"]["interesting"] == ["boat", "dog"]
    assert merged["objects"]["confidence"] == 30


def test_script_total_duration_becomes_the_budget(tmp_path):
    merged = apply_script_to_config(_FakeScript([_FakeBeat()], target=95.0), {})

    assert merged["highlights"]["max_duration"] == 95.0


def test_translation_does_not_mutate_the_caller_config(tmp_path):
    config = {"objects": {"interesting": ["dog"]}}
    original = json.dumps(config, sort_keys=True)

    apply_script_to_config(_FakeScript([_FakeBeat(objects=["boat"])]), config)

    assert json.dumps(config, sort_keys=True) == original
