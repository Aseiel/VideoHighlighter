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
    describe_segment_reading,
    explain_standout,
    summarise_expression_arc,
    summarise_run,
    summarise_standouts,
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


class TestConfidence:
    """Points are identical whether the detector was certain or guessing."""

    PEERS = [1.0, 2.0, 100.0]

    def test_the_detector_confidence_is_stated(self):
        text = describe(_entry(score=100.0, detection_confidence=0.94), self.PEERS)
        assert "its strongest detection at 0.94" in text

    def test_a_weak_detection_is_reported_just_as_plainly(self):
        text = describe(_entry(score=100.0, detection_confidence=0.31), self.PEERS)
        assert "0.31" in text

    def test_an_action_is_named_with_its_confidence(self):
        entry = _entry(score=100.0)
        entry["actions"] = [{"name": "jumping", "confidence": 0.88, "tier": None}]
        assert "jumping recognised at 0.88" in describe(entry, self.PEERS)

    def test_the_confidence_tier_is_carried_through(self):
        entry = _entry(score=100.0)
        entry["actions"] = [{"name": "jumping", "confidence": 0.9, "tier": "bonus"}]
        assert "(bonus)" in describe(entry, self.PEERS)

    def test_the_strongest_action_is_the_one_reported(self):
        entry = _entry(score=100.0)
        entry["actions"] = [{"name": "weak", "confidence": 0.4, "tier": None},
                            {"name": "strong", "confidence": 0.9, "tier": None}]
        text = describe(entry, self.PEERS)
        assert "strong" in text and "weak recognised" not in text

    def test_an_action_is_preferred_over_a_box(self):
        """It carries a tier and a name; a box confidence carries neither."""
        entry = _entry(score=100.0, detection_confidence=0.94)
        entry["actions"] = [{"name": "jumping", "confidence": 0.5, "tier": None}]
        text = describe(entry, self.PEERS)
        assert "jumping" in text and "strongest detection" not in text

    def test_no_confidence_recorded_means_no_claim(self):
        assert "detection at" not in describe(_entry(score=100.0), self.PEERS)


# ---------------------------------------------------------------------------
# The comparative reading: what was on screen, against the rest of the video.
# ---------------------------------------------------------------------------

def _subject(name="dog", **over):
    subject = {
        "name": name,
        "at": 30,
        "frame_share": 4.0,
        "frame_share_percentile": 50.0,
        "stretches": 40,
        "stretch_seconds": 12,
        "enough_samples": True,
        "detections": 400,
        "clip_presence_pct": 90.0,
        "confidence": 0.9,
        "prevalence_pct": 60.0,
    }
    subject.update(over)
    return subject


def _relative(**over):
    relative = {"reference": "person", "at": 30, "ratio": 2.0,
                "percentile": 96.0, "median": 1.0, "seconds_together": 240,
                "stretches": 20, "stretch_seconds": 12, "enough_samples": True}
    relative.update(over)
    return relative


def _expression(**over):
    expression = {"label": "surprise", "at": 30, "confidence": 0.88,
                  "confidence_percentile": 50.0, "stretches": 20,
                  "stretch_seconds": 12, "seconds_read": 10,
                  "clip_share_pct": 20.0, "video_share_pct": 20.0, "lift": 1.0,
                  "label_samples": 40, "enough_samples": True,
                  "video_dominant": "neutral", "video_dominant_share_pct": 80.0}
    expression.update(over)
    return expression


def _compared(subjects=(), expression=None, timestamp="1:30"):
    comparison = {}
    if subjects:
        comparison["subjects"] = list(subjects)
    if expression:
        comparison["expression"] = expression
    return {"timestamp": timestamp, "score": 10.0,
            "measured": {"comparison": comparison}}


class TestSubjectFindings:
    def test_an_ordinary_subject_says_nothing_at_all(self):
        """The common case, and the one that keeps the section worth reading."""
        assert explain_standout(_compared([_subject()])) == []

    def test_a_ratio_against_something_in_frame_names_both_and_its_sample(self):
        entry = _compared([_subject(relative=_relative())])
        line = explain_standout(entry)[0]
        assert "2.0× the area of the person" in line
        assert "96% of this video's 12s stretches" in line
        assert "a usual 1.0×" in line

    def test_an_area_ratio_is_never_left_to_read_as_a_length(self):
        """A box with twice the area is about 1.4 times as long, not twice.

        "2.0×" on its own is read as the second thing every time, which turns a
        correct measurement into an overstatement of roughly forty percent. Both
        numbers go in the sentence, each named.
        """
        line = explain_standout(_compared([_subject(relative=_relative())]))[0]
        assert "2.0× the area" in line and "1.4× across" in line

    def test_bare_frame_share_admits_it_cannot_see_the_camera(self):
        """The claim is weaker than it sounds, so the sentence has to say so."""
        entry = _compared([_subject(frame_share_percentile=95.0)])
        line = explain_standout(entry)[0]
        assert "fills 4.0% of the frame" in line
        assert "camera simply moves closer" in line

    def test_a_ratio_is_preferred_over_frame_share_when_both_are_high(self):
        entry = _compared([_subject(frame_share_percentile=95.0,
                                    relative=_relative())])
        line = explain_standout(entry)[0]
        assert "2.0× the area of the person" in line
        assert "fills" not in line

    def test_too_few_detections_means_no_claim(self):
        entry = _compared([_subject(frame_share_percentile=99.0,
                                    stretches=2, enough_samples=False)])
        assert explain_standout(entry) == []

    def test_a_thinly_observed_pairing_makes_no_ratio_claim(self):
        entry = _compared([_subject(
            relative=_relative(percentile=99.0, stretches=2,
                               enough_samples=False))])
        assert explain_standout(entry) == []

    def test_a_rare_class_is_worth_saying_on_its_own(self):
        entry = _compared([_subject(prevalence_pct=3.0)])
        assert "only 3% of the video's detected seconds" in explain_standout(entry)[0]

    def test_rarity_is_not_gated_behind_the_size_comparison(self):
        """The gate that would have made the finding impossible to reach.

        A class the video barely shows has, by definition, few stretches to be
        size-ranked against. Holding rarity to that count means the only classes
        that can be called rare are the ones that are not.
        """
        entry = _compared([_subject(prevalence_pct=2.0, stretches=2,
                                    enough_samples=False)])
        assert "only 2% of the video's detected seconds" in explain_standout(entry)[0]

    def test_a_class_seen_only_twice_is_unconfirmed_not_rare(self):
        entry = _compared([_subject(prevalence_pct=1.0, detections=2)])
        assert explain_standout(entry) == []

    def test_a_flicker_carries_the_share_of_the_clip_it_held(self):
        entry = _compared([_subject(relative=_relative(),
                                    clip_presence_pct=6.0)])
        assert "present for only 6% of the clip" in explain_standout(entry)[0]

    def test_a_size_claim_on_a_doubtful_box_carries_the_number(self):
        entry = _compared([_subject(relative=_relative(), confidence=0.38)])
        assert "on a 0.38 detection" in explain_standout(entry)[0]


class TestExpressionFindings:
    def test_an_unremarkable_expression_says_nothing(self):
        assert explain_standout(_compared(expression=_expression())) == []

    def test_a_label_far_above_its_video_rate_is_reported_with_both_shares(self):
        entry = _compared(expression=_expression(clip_share_pct=80.0,
                                                 video_share_pct=8.0, lift=10.0))
        line = explain_standout(entry)[0]
        assert "80% of this clip" in line and "8% of the video" in line
        assert "10.0×" in line

    def test_the_claim_is_about_the_classifier_not_the_person(self):
        """Five coarse classes and no notion of intensity cannot support more.

        The wording is the safeguard: a reader told what the model reported can
        discount it, a reader told what someone felt cannot.
        """
        entry = _compared(expression=_expression(lift=5.0, clip_share_pct=50.0,
                                                 video_share_pct=10.0))
        line = explain_standout(entry)[0]
        assert line.startswith("The expression classifier read")
        assert " is surprised" not in line and " was surprised" not in line

    def test_a_strong_reading_is_ranked_against_that_label_alone(self):
        entry = _compared(expression=_expression(confidence_percentile=97.0))
        line = explain_standout(entry)[0]
        assert "stronger than in 97% of this video's other 12s stretches" in line

    def test_a_label_the_video_barely_shows_supports_no_claim(self):
        entry = _compared(expression=_expression(lift=20.0, label_samples=2,
                                                 enough_samples=False))
        assert explain_standout(entry) == []

    def test_what_the_video_mostly_reads_as_is_the_context_offered(self):
        entry = _compared(expression=_expression(lift=6.0, clip_share_pct=60.0,
                                                 video_share_pct=10.0))
        assert "mostly neutral (80%)" in explain_standout(entry)[0]


class TestRunLevelStandouts:
    def test_the_clip_each_axis_singled_out_is_named_by_time(self):
        report = {"segments": [
            _compared([_subject()], timestamp="0:10"),
            _compared([_subject(relative=_relative(percentile=99.0))],
                      expression=_expression(lift=9.0), timestamp="4:20"),
        ]}
        line = summarise_standouts(report)
        assert "4:20" in line and "0:10" not in line
        assert "this video only" in line

    def test_nothing_unusual_produces_no_sentence(self):
        report = {"segments": [_compared([_subject()], expression=_expression())]}
        assert summarise_standouts(report) == ""

    def test_a_report_without_comparisons_is_not_an_error(self):
        assert summarise_standouts({"segments": [{"score": 1.0}]}) == ""


# ---------------------------------------------------------------------------
# The video-level expression reading.
# ---------------------------------------------------------------------------

def _analysis(**over):
    analysis = {
        "coverage": {"read_seconds": 900, "duration": 1800.0, "pct": 50.0},
        "labels": {"sad": {"seconds": 500, "share_pct": 55.6,
                           "mean_confidence": 0.71},
                   "happy": {"seconds": 400, "share_pct": 44.4,
                             "mean_confidence": 0.8}},
        "valence": {"mean": -0.2, "mean_all_read": -0.1, "positive_pct": 44.4,
                    "negative_pct": 55.6, "unvalenced_pct": 0.0},
        "stability": {"runs": 20, "mean_run_seconds": 45.0},
        "arc": {"direction": "flat", "change": 0.0, "fit": 0.0,
                "confident": False, "buckets_used": 12},
        "shift": {},
        "episodes": [],
        "dispersion": {},
        "reliability": {"level": "unflagged", "reasons": []},
    }
    analysis.update(over)
    return analysis


class TestExpressionArcProse:
    def test_coverage_leads_because_every_share_below_depends_on_it(self):
        lines = summarise_expression_arc(_analysis())
        assert lines[0].startswith("A face was readable in 50% of the video")

    def test_the_balance_is_given_as_a_ratio_a_person_can_hold(self):
        analysis = _analysis(valence={"mean": -0.5, "mean_all_read": -0.3,
                                      "positive_pct": 10.0, "negative_pct": 52.0,
                                      "unvalenced_pct": 12.0})
        assert any("5.2 to 1 negative" in line
                   for line in summarise_expression_arc(analysis))

    def test_a_near_even_split_is_not_dressed_up_as_a_ratio(self):
        """55 against 44 is not "1.3 to 1", it is a video with no lean."""
        assert any("about evenly split" in line
                   for line in summarise_expression_arc(_analysis()))

    def test_surprise_is_declared_as_counted_on_neither_side(self):
        analysis = _analysis(valence={"mean": -0.2, "mean_all_read": -0.1,
                                      "positive_pct": 30.0, "negative_pct": 50.0,
                                      "unvalenced_pct": 20.0})
        assert any("carries no direction" in line
                   for line in summarise_expression_arc(analysis))

    def test_a_split_is_reported_with_the_time_it_happens(self):
        analysis = _analysis(shift={"at": 605.0, "before": -0.02, "after": -0.33,
                                    "change": -0.31,
                                    "direction": "toward negative"})
        assert any("changes most at 10:05" in line
                   for line in summarise_expression_arc(analysis))

    def test_a_poorly_fitted_slope_is_not_stated_as_a_trend(self):
        """The number that would otherwise be quoted as a finding.

        A slope through a scatter is the easiest misleading sentence this
        module could write, so the weak-fit case has to say so in the sentence
        rather than leave the R² in a field nobody reads.
        """
        analysis = _analysis(arc={"direction": "toward negative", "change": -0.24,
                                  "fit": 0.16, "confident": False,
                                  "buckets_used": 12})
        line = next(l for l in summarise_expression_arc(analysis) if "slope" in l)
        assert "explains only 16%" in line
        assert "not as a trend" in line

    def test_a_well_fitted_slope_is_stated_plainly(self):
        analysis = _analysis(arc={"direction": "toward positive", "change": 0.5,
                                  "fit": 0.8, "confident": True,
                                  "start_valence": -0.2, "end_valence": 0.3,
                                  "buckets_used": 12})
        assert any("drift runs toward positive" in line
                   for line in summarise_expression_arc(analysis))

    def test_negative_stretches_are_listed_as_places_to_look(self):
        analysis = _analysis(episodes=[
            {"start": 2870.0, "end": 2939.0, "seconds": 69.0, "sign": -1,
             "valence": -0.79, "dominant": "sad", "read_seconds": 60},
        ])
        assert any("47:50–48:59" in line
                   for line in summarise_expression_arc(analysis))

    def test_every_reliability_reason_is_surfaced(self):
        analysis = _analysis(reliability={"level": "low",
                                          "reasons": ["only 9% had a face"]})
        assert any(line.startswith("Caution: only 9% had a face")
                   for line in summarise_expression_arc(analysis))

    def test_the_frame_is_always_the_last_word(self):
        """Not a disclaimer bolted on — the accurate description of the data.

        It is last because that is the sentence the reader keeps, and it is
        unconditional because there is no configuration of these numbers that
        makes a label distribution into a record of an experience.
        """
        for analysis in (_analysis(), _analysis(reliability={"level": "low",
                                                            "reasons": ["x"]})):
            last = summarise_expression_arc(analysis)[-1]
            assert "not what anyone felt" in last
            assert "performed expression from a felt one" in last

    def test_nothing_scanned_produces_nothing(self):
        assert summarise_expression_arc({}) == []


class TestSegmentReading:
    def test_a_clip_matching_the_video_is_called_ordinary(self):
        row = {"valence": -0.3, "delta": -0.02, "dominant": "sad",
               "read_seconds": 10}
        assert "in line with the rest" in describe_segment_reading(row, -0.28)

    def test_a_clip_running_against_the_file_is_named_as_such(self):
        row = {"valence": 0.4, "delta": 0.68, "dominant": "happy",
               "read_seconds": 10}
        assert "more positive than the video's own average" in \
            describe_segment_reading(row, -0.28)

    def test_a_clip_with_no_reading_says_nothing(self):
        assert describe_segment_reading({"read_seconds": 0}) == ""


class TestEffectSize:
    """A rank without a magnitude is a true sentence that misleads.

    Classes whose boxes overlap the same region of the frame sit at ratios near
    1.0 to each other by construction, and the largest of those is still the
    largest — 98th percentile on an eight-percent difference. The percentile is
    correct and the sentence it licenses is worthless, so the claim needs both.
    """

    def test_a_top_ranked_but_tiny_difference_is_not_narrated(self):
        entry = _compared([_subject(relative=_relative(
            ratio=1.14, median=1.10, percentile=98.0))])
        assert explain_standout(entry) == []

    def test_a_real_difference_at_the_same_rank_is(self):
        entry = _compared([_subject(relative=_relative(
            ratio=5.60, median=1.10, percentile=98.0))])
        assert "5.6× the area" in explain_standout(entry)[0]

    def test_being_unusually_small_also_counts_as_a_difference(self):
        entry = _compared([_subject(relative=_relative(
            ratio=0.40, median=1.10, percentile=95.0))])
        assert explain_standout(entry) != []

    def test_a_big_ratio_that_is_simply_normal_here_is_not_a_finding(self):
        """8.0x sounds enormous; if the video's median is 8.0 it is the baseline."""
        entry = _compared([_subject(relative=_relative(
            ratio=8.10, median=8.00, percentile=99.0))])
        assert explain_standout(entry) == []


class TestClassConditionedProse:
    def test_a_flat_result_says_so_and_blocks_the_attribution(self):
        """The sentence that stops a file-wide number being read as a cause."""
        analysis = _analysis(by_class=[
            {"name": "dog", "read_seconds": 554, "valence": -0.299,
             "delta": 0.001, "distinguishable": False, "dominant": "sad",
             "shares": {}},
        ])
        line = next(l for l in summarise_expression_arc(analysis)
                    if "No detected class" in l)
        assert "dog +0.00 over 554s" in line
        assert "cannot be attributed" in line

    def test_a_real_difference_is_stated_without_claiming_a_cause(self):
        analysis = _analysis(by_class=[
            {"name": "dog", "read_seconds": 300, "valence": 0.40,
             "delta": 0.50, "distinguishable": True, "dominant": "happy",
             "shares": {}},
        ])
        line = next(l for l in summarise_expression_arc(analysis)
                    if "While dog is on screen" in l)
        assert "more positive" in line
        assert "what caused the difference is not" in line

    def test_no_breakdown_adds_no_lines(self):
        assert not any("No detected class" in l
                       for l in summarise_expression_arc(_analysis()))
