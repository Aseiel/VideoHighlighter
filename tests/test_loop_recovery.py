"""Speech behind a Whisper repetition loop is decoded again and kept."""
from modules.audio import loop_recovery as lr
from modules.audio.transcript_cleanup import clean_segments

LOOP = ", ".join(["Um"] * 40)


def seg(start, end, text):
    return {"start": start, "end": end, "text": text}


def test_plain_speech_is_not_a_loop():
    assert not lr.is_loop(seg(0, 2, "No, no, no, that is mine."))
    assert lr.is_loop(seg(0, 0, LOOP))


def test_a_loop_with_a_hole_behind_it_is_found():
    segments = [seg(0, 3, "Hello."), seg(10, 10, LOOP), seg(40, 43, "Team.")]
    assert lr.swallowed_stretches(segments, 600) == [(1, 2, 10.0, 40.0)]


def test_a_loop_running_into_the_next_line_swallowed_nothing():
    segments = [seg(10, 12, LOOP), seg(12.5, 15, "Right.")]
    assert lr.swallowed_stretches(segments, 600) == []


def test_a_loop_at_the_end_runs_to_the_end_of_the_audio():
    assert lr.swallowed_stretches([seg(50, 50, LOOP)], 80) == [(0, 1, 50.0, 80.0)]


def test_the_hole_is_decoded_again_from_the_first_start_that_does_not_loop():
    segments = [seg(0, 3, "Hello."), seg(10, 10, LOOP), seg(40, 43, "Team.")]
    asked = []

    def redecode(start, end):
        asked.append((start, end))
        if start < 13:                       # still opens on the bad sound
            return [seg(start, start, LOOP)]
        return [seg(start + 1, start + 4, "Are you going to buy anything?"),
                seg(35, 60, "Make it double.")]   # padded past the hole

    logged = []
    out = lr.recover(segments, redecode, 600, logged.append)

    assert asked == [(11.5, 40.0), (13.0, 40.0)]
    texts = [s["text"] for s in clean_segments(out)]
    assert texts == ["Hello.", "Um.", "Are you going to buy anything?",
                     "Make it double.", "Team."]
    assert out[-2]["end"] == 40.0            # clipped to the line that follows
    assert len(logged) == 1 and "0:10" in logged[0]


def test_a_stretch_that_loops_every_time_is_left_alone():
    segments = [seg(10, 10, LOOP), seg(40, 43, "Team.")]
    out = lr.recover(segments, lambda a, b: [seg(a, a, LOOP)], 600)
    assert out == segments


def test_no_loop_means_no_decoding():
    segments = [seg(0, 3, "Hello."), seg(40, 43, "Team.")]

    def redecode(start, end):
        raise AssertionError("nothing to recover")

    assert lr.recover(segments, redecode, 600) == segments
