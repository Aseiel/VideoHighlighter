"""Tests for what it would take to measure something the run has no signal for.

The advice this produces costs the user real time — an afternoon of labelling
is the expensive branch — so the properties worth pinning down are the ones
that would make it waste some:

* a route is never offered without the prerequisite it needs in this run;
* the proxy route, which measures the *talking* rather than the thing, is never
  presented as a way to measure the thing;
* the cheap five-minute test is offered before the expensive route, not after;
* the ranking is over cost and reliability and nothing else — this module is
  never told what was said, and could not use it if it were.

Fixture classes are named after workshop objects, as elsewhere: the repo ships
no vocabulary of its own and nothing here knows one class name from another.
"""
from __future__ import annotations

import pytest

from modules import detection_routes as routes


def _report(classes=("bench_vice", "lathe"), events=(), *, faces=0, actions=0,
            visual=False, speech=True):
    return {
        "vocabulary": {"classes": list(classes), "events": list(events)},
        "chapters": [{"number": 1, "method": "visual" if visual else "shot-length"}],
        "settings": {"detector_activity": {"face": faces, "action": actions}},
        "speech": {"words": 400} if speech else {},
    }


# A build that ships every engine in the catalogue. Passed explicitly wherever
# a test asserts *which* route wins, because the same file ships in editions
# that carry different engines: a test reading the ambient build passes on the
# machine it was written on and fails on the other one, having caught nothing.
# The tests that care about a leaner build say so the same way.
def _every_engine(_module):
    return True


class TestCapabilities:
    def test_they_come_from_what_the_run_produced(self):
        caps = routes.capabilities(_report(faces=900, visual=True))
        assert caps["classes"] == ["bench_vice", "lathe"]
        assert caps["faces"] is True
        assert caps["clip_index"] is True
        assert caps["actions"] is False

    def test_shot_length_chapters_mean_no_index_was_ever_built(self):
        # The chapters are cut on a CLIP index when there is one and fall back
        # to shot length when there is not, so they are the record of it.
        assert routes.capabilities(_report())["clip_index"] is False

    def test_an_empty_report_answers_rather_than_raising(self):
        caps = routes.capabilities({})
        assert caps["classes"] == [] and caps["clip_index"] is False


class TestAvailability:
    def test_composing_needs_something_to_compose(self):
        # One class cannot be arranged against anything, and a rule that names
        # a class this video has no detections for can never fire.
        one = routes.available(routes.capabilities(_report(["bench_vice"])))
        assert "compose" not in [r.id for r in one]

        two = routes.available(routes.capabilities(_report()))
        assert "compose" in [r.id for r in two]

    def test_a_face_route_needs_a_face_scan(self):
        without = routes.available(routes.capabilities(_report()))
        assert "face_category" not in [r.id for r in without]

        with_faces = routes.available(routes.capabilities(_report(faces=400)),
                                      _every_engine)
        assert "face_category" in [r.id for r in with_faces]

    def test_the_proxy_route_needs_a_transcript(self):
        quiet = routes.available(routes.capabilities(_report(speech=False)))
        assert "spoken_marker" not in [r.id for r in quiet]


class TestPick:
    def test_two_routes_are_named_and_they_differ(self):
        picked = routes.pick(_report(), _every_engine)
        assert picked["fastest"]["id"] == "example_category"
        assert picked["strongest"]["id"] == "trained_class"

    def test_the_free_check_is_separate_from_the_two(self):
        # Whether a rule can express the claim is not a question of effort --
        # either the classes arrange into it or nothing will -- and the advisor
        # already answers it for the price of one model call.
        picked = routes.pick(_report(), _every_engine)
        assert picked["first"]["id"] == "compose"
        assert picked["fastest"]["id"] != "compose"
        assert picked["strongest"]["id"] != "compose"

    def test_the_proxy_is_never_one_of_the_picks(self):
        # It measures the talking. Offering it as "the fast way to measure
        # this" is the exact dishonesty the section exists to prevent.
        picked = routes.pick(_report(), _every_engine)
        assert picked["interim"]["id"] == "spoken_marker"
        assert picked["fastest"]["confidence_rank"] > routes.CONFIDENCE_PROXY
        assert picked["strongest"]["confidence_rank"] > routes.CONFIDENCE_PROXY

    def test_the_cheap_test_comes_with_the_expensive_route(self):
        picked = routes.pick(_report(), _every_engine)
        assert picked["probe"]["route"] == "open_vocabulary"
        assert "--query" in picked["probe"]["how"]

    def test_the_picks_are_ordered_by_cost_and_reliability(self):
        picked = routes.pick(_report(faces=400), _every_engine)
        assert picked["fastest"]["effort_rank"] <= \
            picked["strongest"]["effort_rank"]
        assert picked["strongest"]["confidence_rank"] >= \
            picked["fastest"]["confidence_rank"]

    def test_a_run_with_nothing_still_answers(self):
        # No classes, no faces, no transcript: the routes that need nothing are
        # still the answer, and an empty section here would leave the user with
        # a problem and no next step.
        picked = routes.pick({}, _every_engine)
        assert picked["fastest"]["id"] == "example_category"
        assert "first" not in picked and "interim" not in picked

    def test_every_route_states_when_it_will_not_work(self):
        # A recommendation without its failure condition is how somebody spends
        # an afternoon labelling for something a prototype would have caught.
        for route in routes.ROUTES:
            assert route.holds_when and route.fails_when
            assert route.gives and route.effort and route.repeat

    def test_it_is_never_told_what_was_said(self):
        """The signature is the guarantee, so it is asserted rather than assumed.

        Choosing a route by what a spoken word *means* would need a lexicon of
        subject matter, which this repo does not ship. If a claim argument ever
        appears here, that decision has been made somewhere it cannot be seen.
        """
        import inspect

        for name in ("pick", "available", "capabilities"):
            params = inspect.signature(getattr(routes, name)).parameters
            assert "claim" not in params and "text" not in params


class TestWhatThisBuildHas:
    """One file, two editions, and it has to be right in both.

    The editions ship different engines. A route naming one this build does not
    have costs the user a decision before they find out — the same failure as a
    rule naming a class the detector never emits, which is the failure the
    module next door exists to prevent. So availability is *detected*, and
    these tests describe builds other than the one they are running on.
    """

    @staticmethod
    def _without(*missing):
        gone = set(missing)
        return lambda module: (not module) or module not in gone

    def test_a_route_whose_engine_is_absent_is_not_offered(self):
        picked = routes.pick(_report(faces=400),
                             self._without("modules.face_examples"))
        assert "face_category" not in [r["id"] for r in picked["all"]]

    def test_the_leaner_edition_still_gets_two_routes(self):
        # Nothing here may degrade to "there is no way to measure this". The
        # claim is the same claim; what changes is which engines can answer it.
        picked = routes.pick(_report(faces=400), self._without(
            "llm.clip_categories", "llm.owl_detect", "modules.face_examples"))
        assert picked["fastest"]["id"] == "clip_search"
        assert picked["strongest"]["id"] == "trained_class"
        assert picked["first"]["id"] == "compose"
        assert picked["interim"]["id"] == "spoken_marker"
        # Here the cheapest route and the best test are one engine, and the
        # page says so rather than telling the reader to do it first and then
        # again two lines down.
        assert picked["probe"]["same_as_fastest"] is True

    def test_the_probe_falls_back_to_an_engine_that_is_there(self):
        # The control-query test is the most useful thing on the page for
        # somebody about to spend an afternoon labelling, and it must not
        # vanish with the engine it was first written for.
        picked = routes.pick(_report(), self._without("llm.owl_detect"))
        assert picked["probe"]["route"] == "clip_search"
        assert "--query" in picked["probe"]["how"]

    def test_the_probe_prefers_the_engine_that_settles_the_question(self):
        # A detector that cannot see the thing is a stronger answer to "must I
        # train one" than a whole-frame similarity score.
        picked = routes.pick(_report(), _every_engine)
        assert picked["probe"]["route"] == "open_vocabulary"
        assert "same_as_fastest" not in picked["probe"]

    def test_the_record_says_which_engines_answered(self):
        caps = routes.capabilities({})
        assert "compose" in caps["engines"]
        assert set(caps["engines"]) <= {r.id for r in routes.ROUTES}

    def test_this_build_answers_with_whatever_it_ships(self):
        """The one thing that legitimately differs between the editions.

        Which routes exist here is not asserted — that is what the edition
        decides. That there are still two of them is, because a build that
        answered "nothing can measure this" would have turned the section into
        a dead end rather than a next step.
        """
        picked = routes.pick(_report(faces=400))
        assert picked["fastest"]["id"] and picked["strongest"]["id"]
        offered = {r["id"] for r in picked["all"]}
        assert offered <= set(picked["capabilities"]["engines"])
        for key in ("fastest", "strongest", "first", "interim"):
            if picked.get(key):
                assert picked[key]["id"] in offered

    def test_a_route_with_no_engine_named_is_always_available(self):
        # Composition, actions, training and the transcript are in every build.
        for route in routes.ROUTES:
            if not route.module:
                assert routes._installed(route.module) is True


class TestDescribe:
    def test_both_picks_reach_the_prose(self):
        picked = routes.pick(_report(), _every_engine)
        said = " ".join(routes.describe(picked))
        assert "Fastest:" in said and "Most reliable:" in said
        assert picked["fastest"]["name"] in said
        assert picked["strongest"]["name"] in said

    def test_the_free_check_leads(self):
        said = routes.describe(routes.pick(_report(), _every_engine))
        assert said[0].startswith("Costs nothing to rule out")

    def test_nothing_to_say_is_an_empty_list(self):
        assert routes.describe({}) == []
