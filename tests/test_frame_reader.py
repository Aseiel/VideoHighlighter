"""Exact thumbnail frames are read on from the frame before when that is
cheaper than seeking - and only then."""
import threading

import numpy as np

from video_ai_editor.frame_reader import FrameReader


class _Capture:
    """A 10 fps video whose frame number (mod 256) is its pixel value."""

    FPS = 10.0

    def __init__(self, frames=1000):
        self.frames = frames
        self.next = 0                 # the frame the next grab would take
        self.sets = 0
        self.grabs = 0

    def isOpened(self):
        return True

    def get(self, prop):
        return self.FPS

    def set(self, prop, time_ms):
        self.sets += 1
        self.next = int(round(time_ms / 1000.0 * self.FPS))

    def grab(self):
        self.grabs += 1
        self.next += 1
        return self.next <= self.frames

    def retrieve(self):
        return True, np.full((2, 2, 3), (self.next - 1) % 256, dtype=np.uint8)

    def read(self):
        if not self.grab():
            return False, None
        return self.retrieve()

    def release(self):
        pass


class _OpenCV:
    CAP_PROP_FPS, CAP_PROP_POS_MSEC = 5, 0

    def __init__(self):
        self.capture = _Capture()

    def VideoCapture(self, path):
        return self.capture


def _reader():
    opencv = _OpenCV()
    return FrameReader("video.mp4", opencv=opencv), opencv.capture


def _number(frame):
    return int(frame[0, 0, 0])


class TestWhichWayToTheFrame:
    def test_the_first_frame_is_a_seek(self):
        reader, capture = _reader()
        assert _number(reader.frame_at(5000)) == 50
        assert (reader.seeks, reader.read_on, capture.sets) == (1, 0, 1)

    def test_a_frame_just_ahead_is_read_on_to(self):
        reader, capture = _reader()
        reader.frame_at(5000)
        assert _number(reader.frame_at(6000)) == 60      # one second on
        assert (reader.seeks, reader.read_on, capture.sets) == (1, 1, 1)

    def test_a_frame_behind_is_a_seek(self):
        reader, capture = _reader()
        reader.frame_at(5000)
        assert _number(reader.frame_at(2000)) == 20
        assert capture.sets == 2

    def test_a_frame_far_ahead_is_a_seek(self):
        reader, capture = _reader()
        reader.frame_at(5000)
        assert _number(reader.frame_at(90000)) == 900 % 256
        assert capture.sets == 2

    def test_the_same_frame_again_costs_nothing(self):
        reader, capture = _reader()
        first = reader.frame_at(5000)
        grabs = capture.grabs
        assert reader.frame_at(5020) is first            # within the same frame
        assert capture.grabs == grabs and capture.sets == 1

    def test_a_strip_of_close_slots_is_one_seek_and_the_rest_read_on(self):
        reader, capture = _reader()
        got = [_number(reader.frame_at(t)) for t in range(10000, 40000, 1000)]
        assert got == [n % 256 for n in range(100, 400, 10)]
        assert capture.sets == 1 and reader.read_on == 29


class TestItLearnsWhatThisFileCosts:
    def test_cheap_seeks_stop_it_reading_far_ahead(self):
        reader, capture = _reader()
        reader.frame_at(5000)
        reader.seek_ms, reader.skip_ms = 5.0, 1.0        # keyframes close together
        reader.frame_at(6000)                            # 10 frames on: 10 ms > 5 ms
        assert capture.sets == 2

    def test_dear_seeks_let_it_read_further(self):
        reader, capture = _reader()
        reader.frame_at(5000)
        reader.seek_ms, reader.skip_ms = 500.0, 1.0
        reader.frame_at(25000)                           # 200 frames on: 200 ms < 500 ms
        assert capture.sets == 1


class TestTheEndOfTheVideo:
    def test_reading_past_the_end_gives_nothing_and_recovers(self):
        reader, capture = _reader()
        capture.frames = 100
        reader.frame_at(9000)
        assert reader.frame_at(12000) is None
        assert _number(reader.frame_at(5000)) == 50      # and it still works

    def test_closed_gives_nothing(self):
        reader, _ = _reader()
        reader.frame_at(1000)
        reader.close()
        assert reader.frame_at(2000) is None


class TestWaitersAreLetInInOrder:
    def test_urgent_first_then_earliest_in_the_video(self):
        reader, _ = _reader()
        order = []
        reader._turns.enter(0, 0.0)                      # hold the door
        waiting = []
        for priority, time_ms in ((0, 3000.0), (0, 1000.0), (-10, 9000.0), (0, 2000.0)):
            def wait(p=priority, t=time_ms):
                reader._turns.enter(p, t)
                order.append((p, t))
                reader._turns.leave()
            thread = threading.Thread(target=wait)
            thread.start()
            waiting.append(thread)
        while len(reader._turns._waiting) < 4:
            pass
        reader._turns.leave()
        for thread in waiting:
            thread.join(timeout=5)
        assert order == [(-10, 9000.0), (0, 1000.0), (0, 2000.0), (0, 3000.0)]
