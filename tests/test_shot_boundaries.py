"""Frame-accurate shot changes, and the files written from them."""
import numpy as np

from modules.segments import shot_boundaries as sb
from modules.segments import shot_list_export as export


def _frames(*runs):
    """Flat-coloured frames: ``(count, level)`` per run, with a little noise."""
    rng = np.random.default_rng(0)
    out = []
    for count, level in runs:
        block = np.full((count, sb.HEIGHT, sb.WIDTH, 3), level, dtype=np.int16)
        block += rng.integers(-2, 3, block.shape)
        out.append(np.clip(block, 0, 255).astype(np.uint8))
    return np.concatenate(out)


def _steps(frames, block=7):
    differ = sb._Differ()
    for i in range(0, len(frames), block):   # odd blocks: seams everywhere
        differ.feed(frames[i:i + block])
    return differ.result()


def test_a_hard_cut_is_found_on_its_frame():
    step, across = _steps(_frames((40, 30), (40, 200), (40, 90)))
    assert list(sb.cuts_from_steps(step, across, radius=8)) == [40, 80]


def test_blocks_do_not_change_the_numbers():
    frames = _frames((23, 30), (31, 200), (17, 90))
    one = _steps(frames, block=len(frames))
    many = _steps(frames, block=5)
    assert np.allclose(one[0], many[0]) and np.allclose(one[1], many[1])
    assert one[0].size == one[1].size == len(frames)


def test_a_one_frame_flash_is_not_a_cut():
    step, across = _steps(_frames((40, 30), (1, 250), (40, 30)))
    assert list(sb.cuts_from_steps(step, across, radius=8)) == []


def test_steady_noise_is_not_a_cut():
    step, across = _steps(_frames((120, 100)))
    assert list(sb.cuts_from_steps(step, across, radius=8)) == []


def test_a_busy_stretch_does_not_pass_on_ratio():
    step = np.full(100, 0.08)
    step[50] = 0.12                       # largest, but not standing alone
    assert list(sb.cuts_from_steps(step, np.ones(100), radius=8)) == []
    step[50] = 0.40                       # two different pictures
    assert list(sb.cuts_from_steps(step, np.ones(100), radius=8)) == [50]


def _shots(**kw):
    base = dict(video_path="/v/My Film.mkv", fps_num=25, fps_den=1,
                frame_count=250, duration=10.0, cut_frames=(50, 125),
                cut_times=(2.0, 5.0))
    base.update(kw)
    return sb.ShotList(**base)


def test_shots_cover_the_video():
    assert _shots().shots == [(0.0, 2.0), (2.0, 5.0), (5.0, 10.0)]


def test_edl_has_one_cut_only_event_per_shot_from_zero():
    lines = export.edl_text(_shots()).splitlines()
    assert lines[0] == "TITLE: My Film shots"
    assert lines[1] == "FCM: NON-DROP FRAME"
    events = [l for l in lines if l[:3].isdigit()]
    assert events == [
        "001  MYFILM   V     C        00:00:00:00 00:00:02:00 00:00:00:00 00:00:02:00",
        "002  MYFILM   V     C        00:00:02:00 00:00:05:00 00:00:02:00 00:00:05:00",
        "003  MYFILM   V     C        00:00:05:00 00:00:10:00 00:00:05:00 00:00:10:00",
    ]


def test_plain_list_is_one_cut_per_line():
    assert export.plain_text(_shots(cut_times=(2.0, 3725.5))) == (
        "00:00:02.000\n01:02:05.500\n")


# --- the learned scorer (modules/segments/shot_model.py) ---------------------
from modules.segments import shot_model as sm


def test_a_one_frame_transition_starts_the_shot_on_the_next_frame():
    scores = np.zeros(100)
    scores[40] = 0.9
    cuts, gradual = sm.cuts_from_scores(scores)
    assert list(cuts) == [41] and gradual == []


def test_a_long_transition_is_gradual_and_cut_at_its_middle():
    scores = np.zeros(100)
    scores[40:50] = 0.8
    cuts, gradual = sm.cuts_from_scores(scores)
    assert list(cuts) == [45] and gradual == [(40, 49)]


def test_fading_in_and_out_are_not_shot_changes():
    scores = np.zeros(100)
    scores[:6] = 0.9
    scores[95:] = 0.9
    cuts, gradual = sm.cuts_from_scores(scores)
    assert list(cuts) == [] and gradual == []


class _Echo(sm.ShotModel):
    """A ShotModel with no network: a frame's score is its own red level, so
    the windowing can be checked frame for frame."""

    def __init__(self):
        self._buffer = np.zeros((0, sm.HEIGHT, sm.WIDTH, 3), dtype=np.uint8)
        self._scores, self._count, self._last = [], 0, None
        self.windows = 0

    def _window(self, frames):
        assert frames.shape == (sm.WINDOW, sm.HEIGHT, sm.WIDTH, 3)
        self.windows += 1
        return frames[sm.CONTEXT:sm.CONTEXT + sm.STRIDE, 0, 0, 0] / 255.0


def _numbered(n):
    frames = np.zeros((n, sm.HEIGHT, sm.WIDTH, 3), dtype=np.uint8)
    frames[:, :, :, 0] = (np.arange(n) % 251)[:, None, None]
    return frames


def test_every_frame_gets_its_own_score_whatever_the_blocks():
    for n in (1, 49, 50, 51, 100, 137, 500):
        frames = _numbered(n)
        model = _Echo()
        for i in range(0, n, 37):
            model.feed(frames[i:i + 37])
        got = model.result()
        assert got.shape == (n,)
        assert np.allclose(got, (np.arange(n) % 251) / 255.0), n
        assert model.windows == -(-n // sm.STRIDE)
