"""What has to still be on disk after a run dies badly.

An analysis cache is derived data, but it is derived over hours — transcript,
object boxes and action sequences for a feature-length video. Two habits were
quietly throwing that away:

* a read-modify-write that treated "the entry could not be read" as "there is
  no entry", and wrote back a stub holding only the keys it had just added;
* rewriting the whole entry with a plain ``open(..., "w")``, which truncates
  the destination first, so anything that stops the process inside that window
  leaves wreckage where a complete cache used to be.

Both end the same way for the user: the app comes back up, the analysis looks
absent, and the only way forward is to run the whole thing again.
"""

from __future__ import annotations

import json

import pytest

from modules.video_cache import (
    VideoAnalysisCache,
    atomic_write_json,
    holds_analysis,
)


@pytest.fixture()
def video(tmp_path):
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"not really a video, but it has a size and an mtime")
    return path


@pytest.fixture()
def cache(tmp_path):
    return VideoAnalysisCache(cache_dir=str(tmp_path / "cache"))


PARAMS = {"sample_rate": 1, "duration": 60.0}


def _expensive_analysis():
    return {
        "transcript": {"segments": [{"start": 0.0, "end": 2.0, "text": "hours of this"}]},
        "objects": [{"timestamp": 3, "objects": ["a"], "count": 1}],
        "object_bboxes": [{"frame": 90, "box": [1, 2, 3, 4]}],
    }


class TestHighlightSaveDoesNotEatTheAnalysis:
    def test_an_unreadable_entry_is_left_alone(self, cache, video):
        cache.save(str(video), _expensive_analysis(), params=PARAMS)
        path = cache._get_analysis_cache_path_for_signature(
            str(video), cache._make_signature(PARAMS))

        # However it got this way — a half-finished write, a disk error — the
        # entry is there and unreadable, which is the case that used to be
        # indistinguishable from "nothing cached yet".
        path.write_text('{"transcript": {"segm', encoding="utf-8")

        assert cache.save_highlight_segments(
            str(video), {"min_score": 5}, [(1.0, 2.0)], [{}], {},
            analysis_params=PARAMS) is False

        # Still the damaged file, not a tidy stub written over the top of it.
        # Damaged is recoverable by hand; overwritten is not.
        assert path.read_text(encoding="utf-8") == '{"transcript": {"segm'

    def test_an_entry_for_a_changed_video_is_left_alone(self, cache, video):
        cache.save(str(video), _expensive_analysis(), params=PARAMS)
        path = cache._get_analysis_cache_path_for_signature(
            str(video), cache._make_signature(PARAMS))

        # `load` also refuses on a video_hash mismatch. That is a stale entry,
        # not an absent one, and the highlights being saved do not belong to it.
        disk = json.loads(path.read_text(encoding="utf-8"))
        disk["video_hash"] = "a hash from before the file was re-encoded"
        path.write_text(json.dumps(disk), encoding="utf-8")

        assert cache.save_highlight_segments(
            str(video), {"min_score": 5}, [(1.0, 2.0)], [{}], {},
            analysis_params=PARAMS) is False
        assert "transcript" in json.loads(path.read_text(encoding="utf-8"))

    def test_a_readable_entry_keeps_everything_it_had(self, cache, video):
        cache.save(str(video), _expensive_analysis(), params=PARAMS)

        assert cache.save_highlight_segments(
            str(video), {"min_score": 5}, [(1.0, 2.0)], [{}], {},
            analysis_params=PARAMS) is True

        loaded = cache.load(str(video), params=PARAMS)
        assert loaded["transcript"]["segments"][0]["text"] == "hours of this"
        assert loaded["object_bboxes"] == [{"frame": 90, "box": [1, 2, 3, 4]}]
        assert loaded["highlight_segments"] == [(1.0, 2.0)] or \
               loaded["highlight_segments"] == [[1.0, 2.0]]

    def test_nothing_cached_yet_still_seeds_an_entry(self, cache, video):
        # The refusal must not cost the ordinary case: a highlight run before
        # any analysis has been cached still gets to record itself.
        assert cache.save_highlight_segments(
            str(video), {"min_score": 5}, [(1.0, 2.0)], [{}], {},
            analysis_params=PARAMS) is True

        # Read back through the history reader rather than `load`. The entry
        # holds highlights and no analysis, so `load` is right to refuse it —
        # handing it to the pipeline is what makes a run look lost. The
        # highlights are still there for the reader that wants them.
        assert cache.get_highlight_history(str(video), analysis_params=PARAMS)
        assert cache.load(str(video), params=PARAMS) is None


class TestStubsDoNotStandInForARun:
    def test_a_single_signal_stub_is_not_served_as_a_finished_run(self, cache, video):
        # What older builds left on disk: one on-demand signal, stamped
        # complete. `load` returning this makes pipeline.py set `using_cache`
        # and skip transcript, objects and actions — the run looks lost.
        path = cache._get_cache_path(str(video))
        atomic_write_json(path, {
            "video_path": str(video),
            "video_hash": cache._get_video_hash(str(video)),
            "cache_complete": True,
            "objects": [{"timestamp": 3, "objects": ["a"], "count": 1}],
            "composed_event_names": ["something"],
        })

        assert cache.load(str(video)) is None

    def test_a_real_analysis_is_still_served(self, cache, video):
        cache.save(str(video), _expensive_analysis(), params=PARAMS)
        assert cache.load(str(video), params=PARAMS) is not None

    def test_an_empty_but_genuine_analysis_is_still_served(self, cache, video):
        # A silent clip where nothing was detected is a legitimate result, not a
        # stub. The keys are there; the containers are empty.
        cache.save(str(video), {
            "video_metadata": {"duration": 12.0, "fps": 25.0},
            "transcript": {"segments": [], "language": "en"},
            "objects": [], "actions": [], "scenes": [],
            "motion_events": [], "motion_peaks": [],
        }, params=PARAMS)
        assert cache.load(str(video), params=PARAMS) is not None

    def test_an_older_cache_missing_newer_keys_is_still_served(self, cache, video):
        # `holds_analysis` asks for *any* structural key, not all of them,
        # precisely so upgrading the app does not invalidate work already done.
        cache.save(str(video), {"transcript": {"segments": []}}, params=PARAMS)
        assert cache.load(str(video), params=PARAMS) is not None


class TestHoldsAnalysis:
    def test_a_lone_signal_is_not_an_analysis(self):
        assert holds_analysis({"objects": [], "composed_event_names": []}) is False
        assert holds_analysis({"visual_findings": [1, 2]}) is False
        assert holds_analysis({}) is False

    def test_structural_keys_make_it_one(self):
        assert holds_analysis({"transcript": {}}) is True
        assert holds_analysis({"objects": [], "scenes": []}) is True


class TestAtomicWrite:
    def test_a_failed_write_leaves_the_previous_file_intact(self, tmp_path):
        path = tmp_path / "entry.cache.json"
        atomic_write_json(path, _expensive_analysis())
        before = path.read_text(encoding="utf-8")

        class Unserialisable:
            pass

        with pytest.raises(TypeError):
            atomic_write_json(path, {"objects": Unserialisable()})

        # The point of the tmp-file-then-replace dance: the destination is never
        # the thing being written to, so a write that dies partway through has
        # not touched it.
        assert path.read_text(encoding="utf-8") == before

    def test_it_leaves_no_temp_files_behind(self, tmp_path):
        path = tmp_path / "entry.cache.json"

        class Unserialisable:
            pass

        with pytest.raises(TypeError):
            atomic_write_json(path, {"objects": Unserialisable()})

        assert list(tmp_path.glob("*.tmp")) == []
