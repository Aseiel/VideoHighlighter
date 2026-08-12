"""Tests for modules.script_plan — the script format and its bridge to the cutter.

The traps this file guards against, in order of how much time they cost:

1. **A key that is quietly dropped.** ``durations:`` instead of ``duration:``
   produces a wrong film that looks like an engine bug. Every "must raise" test
   below exists because the alternative is a silent, plausible-looking cut.
2. **repeat losing or reordering clips.** The flattening in
   ``compile_directives`` is the only place the film's shape is decided; a beat
   with ``repeat: 3`` that yields two directives is a shorter film with no error.
3. **A round trip that is not one.** The format's whole promise is "edit the
   file, run again, compare" — so save/load has to return what went in, and
   omitting defaults must not omit meaning.

No files beyond tmp_path, no ffmpeg, no network: this is a parser.
"""

from __future__ import annotations

import pytest

from modules.script_plan import (
    Beat,
    CutDirective,
    Script,
    ScriptError,
    compile_directives,
    example_script,
    load_script,
    parse_script,
    save_script,
    validate_script,
)

MINIMAL = """
title: Test Film
beats:
  - name: Opening
    duration: 10
"""

FULL = """
title: Test Film
music: D:\\music\\track.mp3
snap_to_beat: true
total_duration: 60
beats:
  - name: Establishing
    duration: 12
    order: chronological
    match:
      objects: [alpha, beta]
      keywords: [gamma]
    sources: [CLIP0001.MP4]
  - name: Action
    repeat: 3
    duration: [6, 10]
    match:
      actions: [delta]
"""


def _beat(name="Beat", low=5.0, high=5.0, **kwargs) -> Beat:
    return Beat(name=name, min_duration=low, max_duration=high, **kwargs)


def _script(beats=None, **kwargs) -> Script:
    return Script(title="T", beats=beats if beats is not None else [_beat()], **kwargs)


class TestParse:
    def test_a_single_number_becomes_a_range_whose_ends_agree(self):
        """Downstream must never have to ask which spelling of duration it got."""
        beat = parse_script(MINIMAL).beats[0]
        assert (beat.min_duration, beat.max_duration) == (10.0, 10.0)

    def test_a_range_keeps_both_ends(self):
        beat = parse_script(FULL).beats[1]
        assert (beat.min_duration, beat.max_duration) == (6.0, 10.0)

    def test_the_optional_fields_have_defaults(self):
        script = parse_script(MINIMAL)
        assert script.music == ""
        assert script.snap_to_beat is False
        assert script.total_duration == 0.0
        beat = script.beats[0]
        assert beat.repeat == 1
        assert beat.order == "best_first"
        assert (beat.objects, beat.actions, beat.keywords, beat.sources) == ([], [], [], [])

    def test_a_script_with_no_title_still_has_one(self):
        script = parse_script("beats:\n  - name: A\n    duration: 3\n")
        assert script.title

    def test_every_field_survives_a_full_script(self):
        script = parse_script(FULL)
        assert script.title == "Test Film"
        assert script.music == "D:\\music\\track.mp3"
        assert script.snap_to_beat is True
        assert script.total_duration == 60.0
        assert [b.name for b in script.beats] == ["Establishing", "Action"]
        first, second = script.beats
        assert first.objects == ["alpha", "beta"]
        assert first.keywords == ["gamma"]
        assert first.actions == []
        assert first.sources == ["CLIP0001.MP4"]
        assert first.order == "chronological"
        assert second.repeat == 3
        assert second.actions == ["delta"]

    def test_a_mapping_parses_the_same_as_its_yaml(self):
        """The UI holds a dict, the file holds text; both must land identically."""
        from_dict = parse_script({"title": "Test Film",
                                  "beats": [{"name": "Opening", "duration": 10}]})
        assert from_dict == parse_script(MINIMAL)

    def test_order_is_case_insensitive(self):
        script = parse_script("beats:\n  - name: A\n    duration: 3\n"
                              "    order: Chronological\n")
        assert script.beats[0].order == "chronological"

    def test_an_empty_match_block_is_allowed(self):
        script = parse_script("beats:\n  - name: A\n    duration: 3\n    match:\n")
        assert script.beats[0].has_match_terms is False

    def test_terms_are_stripped_but_otherwise_untouched(self):
        """Terms are the user's own vocabulary; the parser tidies whitespace and
        stops there."""
        script = parse_script("beats:\n  - name: A\n    duration: 3\n"
                              "    match:\n      objects: ['  Two Words  ']\n")
        assert script.beats[0].objects == ["Two Words"]


class TestParseErrors:
    """Each of these must raise. The failure being prevented is not a crash —
    it is a run that completes and quietly ignores what the user wrote."""

    @pytest.mark.parametrize("text, fragment", [
        ("title: x\n", "beats"),
        ("title: x\nbeats: []\n", "at least one beat"),
        ("title: x\nbeats: nope\n", "must be a list"),
        ("beats:\n  - just a string\n", "must be a mapping"),
        ("beats:\n  - duration: 5\n", "no name"),
        ("beats:\n  - name: '  '\n    duration: 5\n", "no name"),
        ("beats:\n  - name: A\n", "no duration"),
        ("beats:\n  - name: A\n    duration: 0\n", "greater than zero"),
        ("beats:\n  - name: A\n    duration: -4\n", "greater than zero"),
        ("beats:\n  - name: A\n    duration: [10, 4]\n", "must not exceed"),
        ("beats:\n  - name: A\n    duration: [1, 2, 3]\n", "range is"),
        ("beats:\n  - name: A\n    duration: soon\n", "must be a number"),
        ("beats:\n  - name: A\n    duration: 5\n    repeat: 0\n", "at least 1"),
        ("beats:\n  - name: A\n    duration: 5\n    repeat: -2\n", "at least 1"),
        ("beats:\n  - name: A\n    duration: 5\n    repeat: 1.5\n", "whole number"),
        ("beats:\n  - name: A\n    duration: 5\n    order: sideways\n", "unknown order"),
        ("beats:\n  - name: A\n    duration: 5\n    match: nope\n", "must be a mapping"),
        ("beats:\n  - name: A\n    duration: 5\n    sources: one.mp4\n", "must be a list"),
        ("snap_to_beat: maybe\nbeats:\n  - name: A\n    duration: 5\n", "true or false"),
        ("total_duration: -5\nbeats:\n  - name: A\n    duration: 5\n", "positive"),
        ("total_duration: soon\nbeats:\n  - name: A\n    duration: 5\n", "must be a number"),
        ("title: [a, b]\nbeats:\n  - name: A\n    duration: 5\n", "must be text"),
    ])
    def test_rejected(self, text, fragment):
        with pytest.raises(ScriptError) as excinfo:
            parse_script(text)
        assert fragment in str(excinfo.value)

    def test_a_match_list_written_as_a_bare_string_is_refused(self):
        """`objects: one` would work today and silently become one nonsense
        term the day it grows a comma."""
        with pytest.raises(ScriptError, match="must be a list"):
            parse_script("beats:\n  - name: A\n    duration: 5\n"
                         "    match:\n      objects: one\n")

    def test_an_unfinished_list_entry_is_refused(self):
        with pytest.raises(ScriptError, match="empty entry"):
            parse_script("beats:\n  - name: A\n    duration: 5\n"
                         "    match:\n      objects:\n        - one\n        -\n")

    def test_an_unknown_top_level_key_stops_the_parse(self):
        with pytest.raises(ScriptError, match="unknown top level key"):
            parse_script("titel: x\nbeats:\n  - name: A\n    duration: 5\n")

    def test_a_typo_in_a_beat_key_stops_the_parse(self):
        """The named failure mode: 'durations:' silently ignored costs an hour."""
        with pytest.raises(ScriptError) as excinfo:
            parse_script("beats:\n  - name: A\n    durations: 5\n")
        message = str(excinfo.value)
        assert "unknown beat key 'durations'" in message
        assert "did you mean 'duration'" in message

    def test_an_unknown_match_key_stops_the_parse(self):
        with pytest.raises(ScriptError, match="unknown match key"):
            parse_script("beats:\n  - name: A\n    duration: 5\n"
                         "    match:\n      objekts: []\n")

    def test_match_terms_at_the_wrong_level_say_where_they_belong(self):
        with pytest.raises(ScriptError) as excinfo:
            parse_script("beats:\n  - name: A\n    duration: 5\n    objects: [x]\n")
        assert "match" in str(excinfo.value)

    @pytest.mark.parametrize("term", ["objects", "actions", "keywords"])
    def test_every_match_term_at_beat_level_says_where_it_belongs(self, term):
        """'actions' is two edits from 'duration', so a spelling checker asked
        first answers a misplaced-but-correctly-spelled key with 'did you mean
        duration?' — sending the one person who knows exactly what they wrote to
        go and stare at an unrelated line."""
        with pytest.raises(ScriptError) as excinfo:
            parse_script(f"beats:\n  - name: A\n    duration: 5\n    {term}: [x]\n")
        message = str(excinfo.value)
        assert "match terms belong under 'match:'" in message
        assert "did you mean" not in message

    def test_a_repeated_key_is_refused_rather_than_last_one_wins(self):
        """PyYAML keeps the last and says nothing — the exact silent drop this
        format exists to refuse."""
        with pytest.raises(ScriptError, match="set twice"):
            parse_script("beats:\n  - name: A\n    duration: 5\n    duration: 9\n")

    def test_broken_yaml_arrives_as_a_script_error(self):
        """Callers guard one exception type; a YAMLError leaking out is a crash."""
        with pytest.raises(ScriptError, match="not valid YAML"):
            parse_script("beats:\n  - name: A\n   duration: [5\n")

    def test_an_empty_document_is_refused(self):
        with pytest.raises(ScriptError, match="empty"):
            parse_script("")

    def test_a_document_that_is_not_a_mapping_is_refused(self):
        with pytest.raises(ScriptError, match="mapping"):
            parse_script("- one\n- two\n")

    def test_a_path_object_is_refused_with_the_right_advice(self, tmp_path):
        with pytest.raises(ScriptError, match="load_script"):
            parse_script(tmp_path / "film.yaml")


class TestNonFiniteNumbers:
    """The one class of number that walks through a range check.

    ``nan <= 0`` and ``nan > high`` are both false, so a NaN duration satisfies
    every guard written against a range and arrives at the engine as a clip
    length — no error, no warning, a run that cuts nothing sensible. Infinity is
    a budget no footage can fill, and neither value can be written back out: a
    saved ``.nan`` is a file this parser refuses to open.
    """

    @pytest.mark.parametrize("written", [".nan", ".inf", "-.inf", "1.0e+400"])
    def test_a_non_finite_duration_is_refused(self, written):
        """'1.0e+400' is infinity spelled as an ordinary decimal — the guard has
        to be on the parsed number, not on how it was typed."""
        with pytest.raises(ScriptError, match="finite"):
            parse_script(f"beats:\n  - name: A\n    duration: {written}\n")

    @pytest.mark.parametrize("written", ["[.nan, 5]", "[5, .inf]"])
    def test_a_non_finite_end_of_a_range_is_refused(self, written):
        with pytest.raises(ScriptError, match="finite"):
            parse_script(f"beats:\n  - name: A\n    duration: {written}\n")

    def test_no_directive_can_carry_a_nan_length(self):
        """The engine reads min_duration/max_duration straight off the
        directive; a NaN there is not recoverable further down."""
        with pytest.raises(ScriptError):
            compile_directives(parse_script(
                "beats:\n  - name: A\n    duration: [.nan, 5]\n    repeat: 2\n"))

    @pytest.mark.parametrize("written", [".nan", ".inf"])
    def test_a_non_finite_total_duration_is_refused(self, written):
        """A NaN total serialises as the bare token NaN, which is not JSON —
        the validate endpoint would hand the browser something it cannot parse."""
        with pytest.raises(ScriptError, match="finite"):
            parse_script(f"total_duration: {written}\n"
                         "beats:\n  - name: A\n    duration: 5\n")

    def test_an_integer_too_large_to_be_a_float_is_refused(self):
        """400 digits is a valid YAML int and not a float at all: float() raises
        OverflowError, which is not a ScriptError and not what callers guard."""
        with pytest.raises(ScriptError, match="too large"):
            parse_script(f"beats:\n  - name: A\n    duration: {'9' * 400}\n")

    def test_saving_a_non_finite_duration_raises_a_script_error(self, tmp_path):
        """A script assembled in code skips the parser. Saving it must fail the
        way the module documents rather than as an OverflowError from int()."""
        script = _script([_beat("A", float("inf"), float("inf"))])
        with pytest.raises(ScriptError, match="finite"):
            save_script(script, str(tmp_path / "film.yaml"))


class TestLineAwareMessages:
    def test_the_line_is_reported(self):
        text = "title: x\nbeats:\n  - name: A\n    duration: 5\n    durations: 5\n"
        with pytest.raises(ScriptError, match=r"line 5"):
            parse_script(text)

    def test_the_line_points_at_the_offending_beat_not_the_first_one(self):
        """Eight similar beats is where a line number stops being a nicety."""
        text = ("beats:\n"
                "  - name: A\n"
                "    duration: 5\n"
                "  - name: B\n"
                "    duration: 5\n"
                "    repeet: 2\n")
        with pytest.raises(ScriptError, match=r"line 6"):
            parse_script(text)

    def test_a_mapping_input_does_not_invent_a_line(self):
        """No source, no line: a made-up position is worse than none."""
        with pytest.raises(ScriptError) as excinfo:
            parse_script({"beats": [{"name": "A", "durations": 5}]})
        assert "line" not in str(excinfo.value)


class TestDirectives:
    def test_repeat_becomes_that_many_directives(self):
        directives = compile_directives(parse_script(FULL))
        assert len(directives) == 4
        assert [d.beat_name for d in directives] == [
            "Establishing", "Action", "Action", "Action"]

    def test_the_index_counts_within_the_beat(self):
        directives = compile_directives(parse_script(FULL))
        assert [d.index for d in directives] == [0, 0, 1, 2]

    def test_directives_are_in_script_order(self):
        script = _script([_beat("First"), _beat("Second"), _beat("Third")])
        assert [d.beat_name for d in compile_directives(script)] == [
            "First", "Second", "Third"]

    def test_a_repeated_beats_constraints_are_carried_to_every_clip(self):
        directives = compile_directives(parse_script(FULL))[1:]
        for directive in directives:
            assert (directive.min_duration, directive.max_duration) == (6.0, 10.0)
            assert directive.actions == ["delta"]

    def test_term_lists_are_copies_not_shared_references(self):
        """One clip narrowed in place must not narrow its siblings."""
        script = _script([_beat("A", repeat=2, objects=["alpha", "beta"])])
        first, second = compile_directives(script)
        first.objects.remove("beta")
        assert second.objects == ["alpha", "beta"]

    def test_a_directive_with_no_terms_says_so(self):
        directive = compile_directives(_script())[0]
        assert directive.has_match_terms is False

    def test_clip_count_matches_the_directives_produced(self):
        script = parse_script(FULL)
        assert script.clip_count == len(compile_directives(script))


class TestAcceptsSource:
    def test_no_sources_allows_everything(self):
        directive = CutDirective("A", 0, 5.0, 5.0)
        assert directive.accepts_source(r"D:\footage\anything.MP4") is True

    def test_a_named_source_matches_by_base_name(self):
        directive = CutDirective("A", 0, 5.0, 5.0, sources=["CLIP0001.MP4"])
        assert directive.accepts_source(r"D:\footage\CLIP0001.MP4") is True

    def test_the_comparison_ignores_case(self):
        """The same file arrives spelled either way on Windows."""
        directive = CutDirective("A", 0, 5.0, 5.0, sources=["clip0001.mp4"])
        assert directive.accepts_source(r"D:\footage\CLIP0001.MP4") is True

    def test_an_unlisted_source_is_rejected(self):
        directive = CutDirective("A", 0, 5.0, 5.0, sources=["CLIP0001.MP4"])
        assert directive.accepts_source(r"D:\footage\CLIP0002.MP4") is False

    def test_a_source_written_as_a_full_path_still_matches(self):
        directive = CutDirective("A", 0, 5.0, 5.0,
                                 sources=[r"E:\old\CLIP0001.MP4"])
        assert directive.accepts_source(r"D:\footage\CLIP0001.MP4") is True


class TestTargetDuration:
    def test_an_explicit_total_wins(self):
        assert parse_script(FULL).target_duration == 60.0

    def test_without_one_the_target_is_the_sum_of_the_maxima(self):
        """Beat maxima times repeat: the most the script asks for, so the
        selector is never short of the length the script describes."""
        script = parse_script(
            "beats:\n"
            "  - name: A\n    duration: 12\n"
            "  - name: B\n    duration: [6, 10]\n    repeat: 3\n")
        assert script.target_duration == 42.0

    def test_clip_count_sums_the_repeats(self):
        script = _script([_beat("A", repeat=3), _beat("B"), _beat("C", repeat=2)])
        assert script.clip_count == 6


class TestRoundTrip:
    def test_a_full_script_survives_save_and_load(self, tmp_path):
        original = parse_script(FULL)
        path = save_script(original, str(tmp_path / "film.yaml"))
        assert load_script(path, log_fn=lambda _m: None) == original

    def test_the_saved_file_reparses_to_the_same_directives(self, tmp_path):
        original = parse_script(FULL)
        path = save_script(original, str(tmp_path / "film.yaml"))
        reloaded = load_script(path, log_fn=lambda _m: None)
        assert compile_directives(reloaded) == compile_directives(original)

    def test_defaults_are_not_written_back_out(self, tmp_path):
        """A saved script has to read like one a person would write; restating
        repeat: 1 on every beat buries the lines that differ."""
        path = save_script(parse_script(MINIMAL), str(tmp_path / "film.yaml"))
        text = (tmp_path / "film.yaml").read_text(encoding="utf-8")
        assert "repeat" not in text
        assert "order" not in text
        assert "match" not in text
        assert "music" not in text
        assert "snap_to_beat" not in text
        assert "total_duration" not in text
        assert load_script(path, log_fn=lambda _m: None) == parse_script(MINIMAL)

    def test_a_duration_range_stays_on_one_line(self, tmp_path):
        """Four lines per range is materially harder to hand-edit, which is the
        one property this format is for."""
        save_script(parse_script(FULL), str(tmp_path / "film.yaml"))
        text = (tmp_path / "film.yaml").read_text(encoding="utf-8")
        assert "duration: [6, 10]" in text

    def test_whole_seconds_are_written_without_a_decimal_point(self, tmp_path):
        save_script(_script([_beat("A", 8.0, 8.0)]), str(tmp_path / "film.yaml"))
        text = (tmp_path / "film.yaml").read_text(encoding="utf-8")
        assert "duration: 8\n" in text

    def test_fractional_seconds_survive(self, tmp_path):
        original = _script([_beat("A", 2.5, 7.25)])
        path = save_script(original, str(tmp_path / "film.yaml"))
        assert load_script(path, log_fn=lambda _m: None).beats[0] == original.beats[0]

    def test_a_duration_finer_than_a_millisecond_survives(self, tmp_path):
        """2.5 and 7.25 are exact at three decimals and so prove nothing about
        rounding. Saving 8.3333 as 8.333 makes 'edit, run, compare' compare two
        different films, and the drift is invisible in the diff the format
        exists to produce."""
        original = _script([_beat("A", 8.3333, 12.00075)])
        path = save_script(original, str(tmp_path / "film.yaml"))
        assert load_script(path, log_fn=lambda _m: None) == original

    def test_a_duration_below_the_old_rounding_step_is_still_loadable(self, tmp_path):
        """Rounding to three decimals sent anything under half a millisecond to
        0.0 — a length parse_script accepts, save_script writes, and load_script
        then refuses. A writer must not emit a file its own reader rejects."""
        original = _script([_beat("A", 0.0004, 0.0004)])
        path = save_script(original, str(tmp_path / "film.yaml"))
        assert load_script(path, log_fn=lambda _m: None) == original

    def test_a_non_ascii_title_survives(self, tmp_path):
        original = Script(title="Zażółć gęślą jaźń", beats=[_beat("A")])
        path = save_script(original, str(tmp_path / "film.yaml"))
        assert load_script(path, log_fn=lambda _m: None).title == original.title

    def test_saving_creates_the_folder(self, tmp_path):
        path = save_script(parse_script(MINIMAL),
                           str(tmp_path / "new" / "film.yaml"))
        assert load_script(path, log_fn=lambda _m: None).beats


class TestLoadScript:
    def test_a_missing_file_raises_a_script_error(self, tmp_path):
        """One exception type for "load this script and cut it"."""
        with pytest.raises(ScriptError):
            load_script(str(tmp_path / "nope.yaml"), log_fn=lambda _m: None)

    def test_the_file_name_is_in_the_message(self, tmp_path):
        path = tmp_path / "second_draft.yaml"
        path.write_text("beats:\n  - name: A\n    durations: 5\n", encoding="utf-8")
        with pytest.raises(ScriptError, match="second_draft.yaml"):
            load_script(str(path), log_fn=lambda _m: None)

    @pytest.mark.parametrize("encoding", ["utf-16", "cp1250"])
    def test_a_file_in_the_wrong_encoding_raises_a_script_error(self, tmp_path,
                                                                encoding):
        """Notepad's 'Unicode' option writes UTF-16 and a Polish title typed in a
        legacy editor arrives as cp1250. Both raise UnicodeDecodeError, which is
        a ValueError and not an OSError, so it slips past the handler and out of
        load_script as the wrong type — the one exception the caller guards is
        the whole promise of this function."""
        path = tmp_path / "draft.yaml"
        path.write_bytes(
            "title: Zażółć\nbeats:\n  - name: A\n    duration: 5\n".encode(encoding))
        with pytest.raises(ScriptError, match="draft.yaml"):
            load_script(str(path), log_fn=lambda _m: None)

    def test_loading_logs_a_summary_and_the_warnings(self, tmp_path):
        path = tmp_path / "film.yaml"
        path.write_text(MINIMAL, encoding="utf-8")
        lines: list[str] = []
        load_script(str(path), log_fn=lines.append)
        assert any(line.startswith("📜") for line in lines)
        assert any(line.startswith("⚠️") for line in lines)


class TestValidate:
    def test_a_complete_script_warns_about_nothing(self, tmp_path):
        music = tmp_path / "track.mp3"
        music.write_bytes(b"")
        script = _script([_beat("A", objects=["alpha"])],
                         music=str(music), snap_to_beat=True)
        assert validate_script(script) == []

    def test_a_beat_that_matches_nothing_is_reported(self):
        warnings = validate_script(_script([_beat("Establishing")]))
        assert any("Establishing" in w and "matches nothing" in w for w in warnings)

    def test_a_beat_restricted_to_sources_is_not_reported(self):
        """"Anything from this one file" is a real instruction, not an omission."""
        script = _script([_beat("A", sources=["CLIP0001.MP4"])])
        assert not any("matches nothing" in w for w in validate_script(script))

    def test_snapping_without_music_is_reported(self):
        script = _script([_beat("A", objects=["alpha"])], snap_to_beat=True)
        assert any("snap_to_beat" in w for w in validate_script(script))

    def test_a_missing_music_file_is_reported(self, tmp_path):
        script = _script([_beat("A", objects=["alpha"])],
                         music=str(tmp_path / "gone.mp3"))
        assert any("music file not found" in w for w in validate_script(script))

    def test_duplicate_beat_names_are_reported(self):
        script = _script([_beat("Action", objects=["a"]), _beat("Action", objects=["b"])])
        assert any("two beats are called" in w for w in validate_script(script))

    def test_a_target_the_beats_cannot_fill_is_reported(self):
        script = _script([_beat("A", 5.0, 5.0, objects=["a"])], total_duration=100.0)
        assert any("come up short" in w for w in validate_script(script))

    def test_a_target_smaller_than_the_beats_is_reported(self):
        script = _script([_beat("A", 50.0, 60.0, objects=["a"])], total_duration=10.0)
        assert any("cut short" in w for w in validate_script(script))

    def test_warnings_never_stop_a_script_compiling(self):
        """A judgement must not be the reason a render does not happen."""
        script = _script([_beat("A")], snap_to_beat=True)
        assert validate_script(script)
        assert len(compile_directives(script)) == 1


class TestExampleScript:
    def test_it_parses(self):
        script = parse_script(example_script())
        assert len(script.beats) >= 3

    def test_it_ships_no_vocabulary_of_its_own(self):
        """Content neutrality: every match term is the user's, supplied at
        runtime. A starter list here would be an opinion the repo must not hold."""
        for beat in parse_script(example_script()).beats:
            assert beat.objects == []
            assert beat.actions == []
            assert beat.keywords == []
            assert beat.sources == []

    def test_it_demonstrates_both_duration_spellings(self):
        beats = parse_script(example_script()).beats
        assert any(b.min_duration == b.max_duration for b in beats)
        assert any(b.min_duration < b.max_duration for b in beats)

    def test_it_survives_a_round_trip(self, tmp_path):
        original = parse_script(example_script())
        path = save_script(original, str(tmp_path / "from_template.yaml"))
        assert load_script(path, log_fn=lambda _m: None) == original

    def test_it_compiles_to_directives(self):
        assert compile_directives(parse_script(example_script()))
