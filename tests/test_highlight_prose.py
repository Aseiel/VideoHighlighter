"""Tests for `modules.highlight_prose` — measurements turned into a sentence.

The property worth protecting is restraint. A sentence may only say what a
measurement supports, and an ordinary moment has to read as ordinary — a report
that calls every clip outstanding carries exactly as much information as one
that says nothing at all.
"""

from __future__ import annotations

from modules.highlight_prose import (
    EXCEPTIONAL,
    describe,
    describe_all,
    summarise_run,
)


def _entry(percentile=50.0, breakdown=None, present=None, score=10.0, **measured):
    m = {"score_percentile": percentile}
    m.update(measured)
    return {
        "breakdown": breakdown or {},
        "signals_present": present or [],
        "measured": m,
        "score": score,
        "start": 0.0,
        "end": 10.0,
    }


# Peers stand in for the rest of the cut: standing is a comparison with them,
# not with the whole video, which a selected clip always tops by construction.
WEAK, MID, STRONG = [1.0, 2.0, 3.0, 4.0, 20.0], 10.0, 100.0


class TestStanding:
    def test_the_best_clip_in_the_cut_is_named(self):
        peers = [1.0, 2.0, 3.0, 4.0, 100.0]
        assert describe(_entry(score=100.0), peers).startswith(
            "The strongest clip in this highlight")

    def test_the_weakest_is_not_flattered(self):
        peers = [1.0, 2.0, 3.0, 4.0, 100.0]
        text = describe(_entry(score=1.0, breakdown={"object": 5.0},
                               present=["object"]), peers)
        assert "weaker clips" in text
        assert "strongest" not in text

    def test_ranking_is_against_the_cut_not_the_video(self):
        """A kept clip tops its own video by construction; saying so is empty."""
        entry = _entry(percentile=99.0, score=1.0)
        assert "strongest" not in describe(entry, [1.0, 50.0, 90.0, 100.0])

    def test_without_peers_no_ranking_is_claimed(self):
        text = describe(_entry(percentile=99.0, breakdown={"object": 5.0},
                               present=["object"]))
        assert "strongest" not in text and "weaker" not in text

    def test_a_cut_where_everything_tied_says_so(self):
        assert "same as every other clip" in describe(
            _entry(score=15.0), [15.0, 15.0, 15.0, 15.0])

    def test_too_few_clips_to_rank(self):
        """No ranking claim — but the share still leads the sentence."""
        text = describe(_entry(score=5.0, percentile=50.0), [5.0, 6.0])
        assert "strongest" not in text and "weaker" not in text
        assert text.startswith("Outscored 50% of the video")


class TestEvidence:
    def test_signals_are_named_the_way_a_person_would(self):
        text = describe(_entry(percentile=50.0,
                               breakdown={"audio": 4.0, "face": 3.0},
                               present=["audio", "face"]))
        assert "a rise in sound" in text and "a facial expression" in text
        assert "audio_peak_points" not in text

    def test_a_single_signal_is_called_out_as_alone(self):
        text = describe(_entry(percentile=50.0, breakdown={"object": 5.0},
                               present=["object"]))
        assert "alone" in text

    def test_several_signals_are_not_called_alone(self):
        text = describe(_entry(percentile=50.0,
                               breakdown={"object": 5.0, "audio": 4.0},
                               present=["object", "audio"]))
        assert "alone" not in text


class TestLoudness:
    def test_a_loud_moment_says_so_with_the_figure(self):
        text = describe(_entry(score=100.0, loudness_percentile=99.0,
                               loudness_dbfs=-1.0), [1.0, 2.0, 100.0])
        assert "loudest points" in text and "-1 dBFS" in text

    def test_an_averagely_loud_moment_does_not_claim_to_be_loud(self):
        text = describe(_entry(score=100.0, loudness_percentile=50.0,
                               loudness_dbfs=-20.0), [1.0, 2.0, 100.0])
        assert "loudest" not in text

    def test_no_audio_measurement_means_no_claim_about_sound(self):
        assert "dBFS" not in describe(_entry(score=100.0), [1.0, 2.0, 100.0])


class TestAgreement:
    def test_coinciding_signals_are_the_headline(self):
        text = describe(_entry(score=100.0, present=["audio", "object"],
                               signals_coincide=True,
                               signal_spread_seconds=0.0), [1.0, 2.0, 100.0])
        assert "landing on the same second" in text
        assert "a rise in sound" in text, "name the signals, not just the count"

    def test_a_gap_is_reported_as_a_gap(self):
        text = describe(_entry(score=100.0, present=["audio", "object"],
                               signals_coincide=True,
                               signal_spread_seconds=3.0), [1.0, 2.0, 100.0])
        assert "within 3s" in text

    def test_signals_that_did_not_coincide_are_not_claimed_to_have(self):
        text = describe(_entry(score=100.0, present=["audio", "object"],
                               signals_coincide=False,
                               signal_spread_seconds=20.0), [1.0, 2.0, 100.0])
        assert "not at the same instant" in text

    def test_one_signal_makes_no_agreement_claim(self):
        text = describe(_entry(score=100.0, present=["object"]), [1.0, 2.0, 100.0])
        assert "landing on the same second" not in text


class TestSentenceShape:
    def test_it_is_one_sentence(self):
        text = describe(_entry(score=100.0, present=["audio", "object"],
                               signals_coincide=True, signal_spread_seconds=0.0,
                               loudness_percentile=99.0, loudness_dbfs=-2.0), [1.0, 2.0, 100.0])
        assert text.count(".") == 1 and text.endswith(".")

    def test_an_empty_entry_produces_nothing_rather_than_a_guess(self):
        assert describe({"measured": {}, "breakdown": {}}) == ""


class TestRunSummary:
    def _report(self, scores, percentiles=None, duration=600.0):
        percentiles = percentiles or [50.0] * len(scores)
        return {
            "video": {"duration": duration},
            "segments": [
                {"score": s, "start": i * 60.0, "end": i * 60.0 + 10.0,
                 "measured": {"score_percentile": p}}
                for i, (s, p) in enumerate(zip(scores, percentiles))
            ],
        }

    def test_an_all_tied_run_says_the_ranking_had_nothing_to_work_with(self):
        text = summarise_run(self._report([15.0, 15.0, 15.0, 15.0]))
        assert "scored identically" in text
        assert "arbitrary" in text

    def test_a_varied_run_reports_the_spread(self):
        text = summarise_run(self._report([20.0, 10.0, 5.0],
                                          percentiles=[99.0, 95.0, 20.0]))
        assert "above the weakest" in text

    def test_it_does_not_claim_clips_were_exceptional(self):
        """Every kept clip beat the video — saying so tells the reader nothing."""
        text = summarise_run(self._report([20.0, 10.0, 5.0],
                                          percentiles=[99.0, 99.0, 99.0]))
        assert "strongest in the video" not in text

    def test_describe_all_ranks_each_against_the_others(self):
        rep = self._report([100.0, 50.0, 1.0, 2.0, 3.0])
        lines = describe_all(rep)
        assert "strongest clip" in lines[0]
        assert "weaker clips" in lines[2]

    def test_it_reports_how_much_of_the_video_was_drawn_from(self):
        assert "%" in summarise_run(self._report([20.0, 10.0],
                                                 percentiles=[99.0, 20.0]))

    def test_nothing_selected_says_so(self):
        assert summarise_run({"segments": []}) == "Nothing was selected."

    def test_two_identical_clips_are_not_called_a_tied_run(self):
        """Two is too few to conclude the ranking failed."""
        assert "identically" not in summarise_run(self._report([15.0, 15.0]))


class TestTies:
    """A tie is information about the scoring, not a rank for the clip."""

    def test_a_clip_tied_with_most_of_the_cut_is_not_called_weaker(self):
        """The reported bug: 15 clips at 15.0 and one at 10.0 had the fifteen
        top scorers described as the ones that scraped in."""
        peers = [15.0] * 15 + [10.0]
        text = describe(_entry(score=15.0), peers)
        assert "weaker" not in text
        assert "same as most of the other clips" in text

    def test_the_genuinely_lower_clip_is_still_marked(self):
        peers = [15.0] * 15 + [10.0]
        assert "weaker" in describe(_entry(score=10.0), peers)

    def test_a_small_tie_group_is_still_ranked(self):
        peers = [1.0, 2.0, 3.0, 10.0, 10.0]
        assert "same as most" not in describe(_entry(score=10.0), peers)

    def test_an_ordinary_clip_is_not_called_weak(self):
        peers = [1.0, 5.0, 10.0, 15.0, 20.0]
        assert "weaker" not in describe(_entry(score=10.0), peers)


class TestVideoShare:
    """The number people look for first belongs in the sentence."""

    def test_every_clip_states_what_it_outscored(self):
        peers = [15.0] * 15 + [10.0]
        for score in (15.0, 10.0):
            text = describe(_entry(score=score, percentile=89.0), peers)
            assert "outscored 89% of the video" in text

    def test_it_is_there_even_when_nothing_else_is(self):
        text = describe(_entry(score=15.0, percentile=94.0), [15.0] * 4)
        assert "outscored 94% of the video" in text

    def test_a_low_share_is_reported_just_as_plainly(self):
        """Much of the video scoring comparably is worth knowing too."""
        peers = [15.0] * 15 + [10.0]
        assert "outscored 31% of the video" in describe(
            _entry(score=15.0, percentile=31.0), peers)

    def test_a_record_without_the_measurement_makes_no_claim(self):
        entry = {"breakdown": {"object": 5.0}, "signals_present": ["object"],
                 "measured": {}, "score": 5.0}
        assert "outscored" not in describe(entry, [5.0, 6.0, 7.0])

    def test_it_leads_the_evidence(self):
        peers = [1.0, 2.0, 100.0]
        text = describe(_entry(score=100.0, percentile=97.0,
                               breakdown={"object": 5.0}, present=["object"]),
                        peers)
        assert text.index("outscored") < text.index("what was on screen")
