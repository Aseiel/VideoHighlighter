"""How long training will take, said before the run. No torch, no Qt."""
import json

import pytest

from training import train_estimate as te


def test_device_kinds():
    assert te.device_kind("xpu") == "xpu"
    assert te.device_kind("xpu:0") == "xpu"
    assert te.device_kind("cuda:1") == "cuda"
    assert te.device_kind("privateuseone:0") == "dml"
    assert te.device_kind("cpu") == "cpu"
    assert te.device_kind("") == "cpu"


def test_a_graphics_card_is_promised_faster_than_the_processor():
    gpu = te.estimate(150, 40, 30, "tiny", "xpu")
    cpu = te.estimate(150, 40, 30, "tiny", "cpu")
    assert gpu.seconds < cpu.seconds
    # A typical first model: a few minutes on the Arc, a quarter hour or less on the CPU.
    assert 120 < gpu.seconds < 600
    assert 400 < cpu.seconds < 1200


def test_more_rounds_and_more_frames_take_longer():
    base = te.estimate(100, 20, 20, "tiny", "cpu")
    assert te.estimate(100, 20, 40, "tiny", "cpu").seconds > base.seconds
    assert te.estimate(300, 20, 20, "tiny", "cpu").seconds > base.seconds
    assert te.estimate(100, 20, 20, "s", "cpu").seconds > base.seconds


def test_measured_fixed_costs_replace_the_defaults(tmp_path):
    store = te.ThroughputStore(str(tmp_path / "s.json"))
    before = te.estimate(200, 50, 10, "tiny", "xpu", store=store)
    store.record_overheads(extract_per_frame=0.02, fixed=12.0)
    after = te.estimate(200, 50, 10, "tiny", "xpu", store=store)
    assert after.breakdown["frames"] == pytest.approx(250 * 0.02)
    assert after.breakdown["setup_and_export"] == pytest.approx(12.0)
    assert after.seconds < before.seconds
    store.record_overheads(fixed=float("nan"))       # ignored
    assert store.overheads()["fixed"] == pytest.approx(12.0)


def test_overheads_do_not_look_like_a_device_row(tmp_path):
    store = te.ThroughputStore(str(tmp_path / "s.json"))
    store.record_overheads(extract_per_frame=0.05)
    assert store.get("cpu", "tiny") is None


def test_the_first_download_is_counted():
    cached = te.estimate(100, 20, 20, "tiny", "xpu", pretrained_cached=True)
    fresh = te.estimate(100, 20, 20, "tiny", "xpu", pretrained_cached=False)
    assert fresh.seconds == pytest.approx(cached.seconds + te.PRETRAINED_DOWNLOAD_SECONDS)


def test_an_unmeasured_device_gets_a_wider_range():
    measured_kind = te.estimate(100, 20, 20, "tiny", "xpu")
    guessed_kind = te.estimate(100, 20, 20, "tiny", "privateuseone:0")
    width = lambda e: (e.high - e.low) / e.seconds  # noqa: E731
    assert width(guessed_kind) > width(measured_kind)


def test_this_computers_own_speed_wins(tmp_path):
    store = te.ThroughputStore(str(tmp_path / "speed.json"))
    guess = te.estimate(200, 50, 30, "tiny", "cpu", store=store)
    assert not guess.measured

    store.record("cpu", "tiny", (416, 416), train_ips=70.0, val_ips=210.0)
    store.save()
    reloaded = te.ThroughputStore(str(tmp_path / "speed.json")).load()
    measured = te.estimate(200, 50, 30, "tiny", "cpu", store=reloaded)
    assert measured.measured
    assert measured.seconds < guess.seconds
    assert "measured on this computer" in measured.sentence("the processor")


def test_speeds_are_blended_not_overwritten(tmp_path):
    store = te.ThroughputStore(str(tmp_path / "s.json"))
    store.record("xpu", "tiny", (416, 416), 30.0, 40.0)
    store.record("xpu", "tiny", (416, 416), 10.0, 20.0)
    train, val = store.get("xpu", "tiny", (416, 416))
    assert train == pytest.approx(20.0)
    assert val == pytest.approx(30.0)


def test_a_nonsense_measurement_is_ignored(tmp_path):
    store = te.ThroughputStore(str(tmp_path / "s.json"))
    store.record("cpu", "tiny", (416, 416), 0.0, 5.0)
    store.record("cpu", "tiny", (416, 416), float("nan"), 5.0)
    assert store.get("cpu", "tiny") is None


def test_an_unreadable_store_is_empty_not_a_crash(tmp_path):
    p = tmp_path / "s.json"
    p.write_text("{not json", encoding="utf-8")
    assert te.ThroughputStore(str(p)).load().rows == {}


def test_the_sentence_warns_when_there_is_too_little_to_learn_from():
    few = te.estimate(10, 2, 30, "tiny", "xpu")
    assert "usually needs" in few.sentence("your Intel Arc")
    enough = te.estimate(100, 20, 30, "tiny", "xpu")
    assert "usually needs" not in enough.sentence("your Intel Arc")


@pytest.mark.parametrize("seconds,words", [
    (20, "under a minute"),
    (70, "about 1 minute"),
    (250, "about 4 minutes"),
    (1260, "about 20 minutes"),
    (5400, "about 1.5 hours"),
    (7200, "about 2 hours"),
])
def test_durations_read_like_a_person_says_them(seconds, words):
    assert te.friendly_duration(seconds) == words


def test_device_phrases():
    assert te.friendly_device("cpu") == "the processor"
    assert te.friendly_device("xpu", "Intel(R) Arc(TM) A750 Graphics") == "your Intel(R) Arc(TM) A750 Graphics"
    assert te.friendly_device("cuda") == "your NVIDIA graphics card"
