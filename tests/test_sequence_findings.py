"""What the sequence findings claim, and — mostly — what they refuse to."""
import pytest

from modules.sequence_findings import findings, summarise


def _report(segments, *, duration=100.0, speech=None, activity=None):
    return {
        "video": {"duration": duration},
        "segments": segments,
        "speech": speech or {},
        "settings": {"detector_activity": activity or {"object": 5, "face": 5}},
    }


def _seg(start, end, *, objects=(), events=(), actions=()):
    return {
        "start": start, "end": end,
        "objects": list(objects), "events": list(events),
        "actions": [{"name": n, "confidence": 0.5} for n in actions],
        "signals_present": ["motion_peak"],
    }


class TestOrder:
    def test_conditions_come_back_in_order_of_first_appearance(self):
        r = _report([_seg(30, 40, objects=["b"]), _seg(10, 20, objects=["a"])])
        names = [c["name"] for c in findings(r)["conditions"]]
        assert names == ["a", "b"]

    def test_interval_is_between_first_appearances(self):
        r = _report([_seg(10, 20, objects=["a"]), _seg(34, 40, objects=["b"])])
        got = {c["name"]: c["since_previous_s"] for c in findings(r)["conditions"]}
        assert got["a"] is None            # nothing precedes the first
        assert got["b"] == pytest.approx(24.0)

    def test_a_condition_seen_twice_keeps_its_first_and_last(self):
        r = _report([_seg(10, 20, objects=["a"]), _seg(50, 60, objects=["a"])])
        (rec,) = findings(r)["conditions"]
        assert (rec["first"], rec["last"], rec["windows"]) == (10.0, 60.0, 2)

    def test_signals_are_not_conditions(self):
        """`motion_peak` describes the scoring, not the material.

        Letting it in produces a sequence in which the detector's own behaviour
        appears as something that happened.
        """
        r = _report([_seg(10, 20, objects=["a"])])
        kinds = {c["kind"] for c in findings(r)["conditions"]}
        assert kinds == {"object"}

    def test_events_and_actions_are_conditions(self):
        r = _report([_seg(10, 20, events=["e"], actions=["act"])])
        assert {c["kind"] for c in findings(r)["conditions"]} == {"event", "action"}


class TestNotEstablished:
    def test_the_permanent_limits_are_always_present(self):
        r = _report([_seg(0, 100, objects=["a"])])
        text = " ".join(findings(r)["not_established"])
        assert "knew, intended, perceived" in text
        assert "what it appears to be" in text
        assert "neither is a cause" in text

    def test_silent_detectors_are_named(self):
        r = _report([_seg(0, 100, objects=["a"])],
                    activity={"object": 5, "keyword": 0, "action": 0})
        text = " ".join(findings(r)["not_established"])
        assert "keyword" in text and "action" in text

    def test_partial_coverage_is_declared(self):
        r = _report([_seg(0, 25, objects=["a"])], duration=100.0)
        out = findings(r)
        assert out["coverage_pct"] == pytest.approx(25.0)
        assert any("was not kept" in x for x in out["not_established"])

    def test_full_coverage_does_not_complain(self):
        r = _report([_seg(0, 100, objects=["a"])], duration=100.0)
        assert not any("was not kept" in x
                       for x in findings(r)["not_established"])


class TestVoiceover:
    """One speaker over nearly all of the runtime is talking *about* the
    material, not in it — and a report that misses this hands the reader a
    narrator's assertions dressed as observations."""

    NARRATED = {"speech_share_pct": 99.2, "speech_seconds": 45.5, "words": 82,
                "segments": 8, "speakers": [{"speaker": "S1", "words": 82}]}

    def test_narration_is_detected_and_flagged(self):
        r = _report([_seg(0, 100, objects=["a"])], speech=self.NARRATED)
        out = findings(r)
        assert out["narration"]["share_pct"] == 99.2
        assert "narration about the material" in out["not_established"][0]

    def test_two_speakers_is_not_narration(self):
        speech = dict(self.NARRATED,
                      speakers=[{"speaker": "S1"}, {"speaker": "S2"}])
        r = _report([_seg(0, 100, objects=["a"])], speech=speech)
        assert findings(r)["narration"] is None

    def test_a_short_remark_is_not_narration(self):
        speech = dict(self.NARRATED, speech_seconds=4.0)
        r = _report([_seg(0, 100, objects=["a"])], speech=speech)
        assert findings(r)["narration"] is None

    def test_sparse_speech_is_not_narration(self):
        speech = dict(self.NARRATED, speech_share_pct=40.0)
        r = _report([_seg(0, 100, objects=["a"])], speech=speech)
        assert findings(r)["narration"] is None

    def test_no_speech_at_all_says_so(self):
        r = _report([_seg(0, 100, objects=["a"])], speech={})
        assert "no speech was transcribed" in findings(r)["not_established"][0]


class TestSummary:
    def test_empty_selection_says_so_rather_than_printing_nothing(self):
        out = summarise(findings(_report([])))
        assert "no sequence to report" in out

    def test_the_limits_are_in_the_text_not_only_the_data(self):
        r = _report([_seg(10, 20, objects=["a"])])
        assert "Not established by this run:" in summarise(findings(r))

    def test_a_malformed_segment_is_skipped_rather_than_fatal(self):
        r = _report([{"start": None, "end": 5, "objects": ["a"]},
                     _seg(10, 20, objects=["b"])])
        assert [c["name"] for c in findings(r)["conditions"]] == ["b"]
