"""Outlines on demand for composition rules (modules/vision/outlines.py).

The pass must trace only what a rule could need, never twice, and keep the
cache's lists aligned; GrabCut must find the shape inside a box. The last is
checked on real pixels where OpenCV is installed (conftest mocks it in CI).
"""
from __future__ import annotations

import os
import sys
import textwrap

import numpy as np
import pytest

from modules.report.analysis_ondemand import _without_classes
from modules.vision import outlines
from video_ai_editor.composition_engine import CompositionEngine

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conftest import real_opencv  # noqa: E402


def _entry(ts, dets):
    return {"timestamp": ts, "objects": [d[0] for d in dets],
            "bboxes": [d[1] for d in dets], "confidences": [0.9] * len(dets)}


PAIRS = [("thing", "holder", 0.0)]


def test_only_detections_whose_boxes_meet_their_partner_are_traced():
    entry = _entry(0.0, [("holder", [0.1, 0.1, 0.3, 0.3]),
                         ("thing", [0.2, 0.2, 0.05, 0.05]),     # meets the holder
                         ("thing", [0.8, 0.8, 0.05, 0.05]),     # nowhere near it
                         ("other", [0.2, 0.2, 0.1, 0.1])])      # not in any rule
    assert outlines.wanted(entry, PAIRS) == {0, 1}
    assert outlines.wanted(entry, [("thing", "holder", 0.5)]) == {0, 1, 2}


class FakeOutliner:
    name = "fake"

    def __init__(self):
        self.calls = []

    def outline(self, frame, boxes_px):
        self.calls.append(len(boxes_px))
        return [[[0.1, 0.1], [0.2, 0.1], [0.2, 0.2]] if i == 0 else None
                for i, _ in enumerate(boxes_px)]


def _frames(video, stamps):
    for ts in stamps:
        yield ts, np.zeros((100, 200, 3), np.uint8)


def test_the_pass_traces_once_and_keeps_lists_aligned():
    cache = [_entry(0.0, [("other", [0, 0, 1, 1]), ("holder", [0.1, 0.1, 0.3, 0.3]),
                          ("thing", [0.2, 0.2, 0.05, 0.05])]),
             _entry(1.0, [("thing", [0.8, 0.8, 0.05, 0.05])])]
    fake = FakeOutliner()
    stats = outlines.add_outlines("v.mp4", cache, PAIRS, outliner=fake, frame_reader=_frames)
    assert stats == {"frames": 1, "traced": 1, "no_outline": 1, "outliner": "fake"}
    contours = cache[0]["contours"]
    assert len(contours) == 3 and contours[0] is None
    assert contours[1] and contours[2] == []          # [] = tried, found nothing
    assert "contours" not in cache[1]

    again = outlines.add_outlines("v.mp4", cache, PAIRS, outliner=fake, frame_reader=_frames)
    assert again["frames"] == 0 and fake.calls == [2]


def test_removing_a_class_keeps_outlines_on_their_own_boxes():
    entry = _entry(0.0, [("a", [0, 0, 0.1, 0.1]), ("b", [0.5, 0.5, 0.1, 0.1])])
    entry["contours"] = [[[0, 0], [0.1, 0], [0.1, 0.1]], [[0.5, 0.5], [0.6, 0.5], [0.6, 0.6]]]
    trimmed = _without_classes([entry], {"a"})[0]
    assert trimmed["objects"] == ["b"]
    assert trimmed["contours"] == [[[0.5, 0.5], [0.6, 0.5], [0.6, 0.6]]]


def test_the_rules_file_says_what_to_trace_and_with_what(tmp_path):
    path = tmp_path / "rules.yaml"
    path.write_text(textwrap.dedent("""
        outliner: sam
        events:
          - name: e
            rules:
              - {source: thing, region: holder, relation: touches, max_gap: 0.02, outline: true}
              - {source: x, region: y}
          - name: off
            enabled: false
            rules:
              - {source: p, region: q, outline: true}
        """), encoding="utf-8")
    engine = CompositionEngine(str(path))
    assert engine.outline_pairs == [("thing", "holder", 0.02)]
    assert engine.outliner == "sam"
    path.write_text("outliner: magic\nevents: []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="outliner"):
        CompositionEngine(str(path))


# --- on real pixels ----------------------------------------------------------------------

cv2_real = real_opencv()
needs_cv2 = pytest.mark.skipif(cv2_real is None, reason="real OpenCV is not installed")


@pytest.fixture
def cv2(monkeypatch):
    monkeypatch.setitem(sys.modules, "cv2", cv2_real)
    return cv2_real


@needs_cv2
def test_grabcut_finds_the_shape_inside_the_box(cv2):
    from modules.rules import shapes

    rng = np.random.default_rng(0)
    frame = cv2.GaussianBlur(rng.integers(60, 140, (360, 640, 3)).astype(np.uint8), (0, 0), 3)
    truth = np.zeros((360, 640), np.uint8)
    # An L: most of its bounding box is empty.
    cv2.fillPoly(truth, [np.array([[200, 60], [260, 60], [260, 220], [420, 220],
                                   [420, 280], [200, 280]])], 1)
    frame[truth == 1] = (170, 70, 120)
    (contour,) = outlines.GrabCutOutliner().outline(frame, [(200, 60, 420, 280)])
    assert contour and len(contour) < 40

    drawn = np.zeros_like(truth)
    cv2.fillPoly(drawn, [np.array([[x * 640, y * 360] for x, y in contour], np.int32)], 1)
    iou = (drawn & truth).sum() / (drawn | truth).sum()
    assert iou > 0.9
    # The empty corner of the box is outside the outline.
    assert not shapes.contains(contour, (400 / 640, 80 / 360))


@needs_cv2
def test_nothing_to_find_leaves_the_box(cv2):
    flat = np.full((200, 200, 3), 128, np.uint8)
    assert outlines.GrabCutOutliner().outline(flat, [(50, 50, 150, 150)]) == [None]


@needs_cv2
def test_masks_become_normalised_outlines(cv2):
    mask = np.zeros((100, 200), np.uint8)
    mask[20:60, 50:150] = 1
    contour = outlines.mask_to_contour(mask)
    xs, ys = [p[0] for p in contour], [p[1] for p in contour]
    assert min(xs) == pytest.approx(0.25) and max(ys) == pytest.approx(0.59)


@needs_cv2
def test_sam_glue_turns_box_prompts_into_outlines(cv2, monkeypatch):
    """The model itself cannot be fetched in CI; this pins what is ours: boxes
    in the shape the transformers SAM processor takes, masks out as outlines,
    a near-empty mask treated as no outline."""
    import contextlib
    import types

    class T:                                   # a tensor, as far as the glue cares
        def __init__(self, a):
            self.a = np.asarray(a)

        def cpu(self):
            return self

        def numpy(self):
            return self.a

        def to(self, device):
            return self

        def is_floating_point(self):
            return self.a.dtype.kind == "f"

        def float(self):
            return T(self.a.astype(np.float32))

        def __getitem__(self, i):
            return T(self.a[i])

        def __iter__(self):
            return (T(row) for row in self.a)

    seen = {}

    class Processor:
        def __call__(self, images, input_boxes, return_tensors):
            seen["boxes"] = input_boxes
            return {"pixel_values": T(0), "original_sizes": T([[100, 200]]),
                    "reshaped_input_sizes": T([[100, 200]])}

        class image_processor:
            @staticmethod
            def post_process_masks(pred, original, reshaped):
                full = np.zeros((2, 1, 100, 200), bool)
                full[0, 0, 10:50, 20:80] = True        # box 1: a real mask
                full[1, 0, 0, 0] = True                # box 2: next to nothing
                return [T(full)]                        # one image: [boxes, 1, H, W]

    class Model:
        def __call__(self, **kw):
            seen["multimask"] = kw.get("multimask_output")
            return types.SimpleNamespace(pred_masks=T(0))

    fake_torch = types.SimpleNamespace(inference_mode=contextlib.nullcontext)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    sam = outlines.SamOutliner()
    sam._model, sam._processor, sam.device = Model(), Processor(), "cpu"
    result = sam.outline(np.zeros((100, 200, 3), np.uint8),
                         [(20, 10, 80, 50), (100, 60, 150, 90)])
    assert seen["boxes"] == [[[20.0, 10.0, 80.0, 50.0], [100.0, 60.0, 150.0, 90.0]]]
    assert seen["multimask"] is False
    assert result[1] is None
    xs = [p[0] for p in result[0]]
    assert min(xs) == pytest.approx(0.1) and max(xs) == pytest.approx(0.395)


class _Tensor:
    """Just enough of a tensor to record what ``to_device`` does to it."""

    def __init__(self, dtype, moves=None):
        self.dtype, self.moves = dtype, list(moves or [])

    def is_floating_point(self):
        return self.dtype.startswith("float")

    def float(self):
        return _Tensor("float32", self.moves)

    def to(self, device):
        return _Tensor(self.dtype, self.moves + [device])


def test_processor_floats_reach_the_device_as_float32():
    """Intel GPUs have no float64: the processor's float64 boxes are cast on
    the CPU, before the move. Sizes for post-processing move unchanged."""
    moved = outlines.to_device({
        "pixel_values": _Tensor("float32"),
        "input_boxes": _Tensor("float64"),
        "original_sizes": _Tensor("float64"),
        "reshaped_input_sizes": _Tensor("int64"),
        "note": "not a tensor",
    }, "xpu")
    assert moved["input_boxes"].dtype == "float32"
    assert moved["pixel_values"].dtype == "float32"
    assert moved["original_sizes"].dtype == "float64"
    assert moved["reshaped_input_sizes"].dtype == "int64"
    assert all(moved[k].moves == ["xpu"] for k in moved if k != "note")
    assert moved["note"] == "not a tensor"


def _torch_with(cuda: bool, xpu=None):
    import types

    fake = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: cuda))
    if xpu is not None:
        fake.xpu = types.SimpleNamespace(is_available=lambda: xpu)
    return fake


@pytest.mark.parametrize("cuda, xpu, expected", [
    (True, True, "cuda"),
    (False, True, "xpu"),
    (False, False, "cpu"),
    (False, None, "cpu"),          # a torch build with no xpu module at all
])
def test_auto_device_finds_intel_gpus_too(cuda, xpu, expected):
    assert outlines.pick_device(_torch_with(cuda, xpu)) == expected


def test_auto_device_is_resolved_on_load(monkeypatch):
    import types

    loaded = {}

    class Loader:
        @classmethod
        def from_pretrained(cls, model_id):
            return cls()

        def to(self, device):
            loaded["device"] = device
            return self

        def eval(self):
            return self

    monkeypatch.setitem(sys.modules, "torch", _torch_with(False, True))
    monkeypatch.setitem(sys.modules, "transformers",
                        types.SimpleNamespace(SamModel=Loader, SamProcessor=Loader))
    sam = outlines.SamOutliner()
    sam._load()
    assert sam.device == "xpu" and loaded["device"] == "xpu"


def test_the_pipeline_path_traces_on_copies_and_reports_it(tmp_path, monkeypatch):
    from modules.rules import compose_events

    path = tmp_path / "rules.yaml"
    path.write_text(textwrap.dedent("""
        events:
          - name: e
            window_secs: 0
            persist_secs: 0
            rules:
              - {source: thing, region: holder, relation: overlaps, min_overlap: 0.9,
                 outline: true}
        """), encoding="utf-8")
    cache = [_entry(float(t), [("holder", [0.0, 0.0, 0.6, 0.6]),
                               ("thing", [0.45, 0.05, 0.1, 0.1])]) for t in range(3)]

    def fake_add(video, boxes, pairs, outliner=None, **kw):
        for frame in boxes:     # the holder is an L; the thing is in its empty corner
            frame["contours"] = [[[0, 0], [0.2, 0], [0.2, 0.4], [0.6, 0.4],
                                  [0.6, 0.6], [0, 0.6]], []]
        return {"frames": len(boxes), "traced": len(boxes), "no_outline": len(boxes),
                "outliner": "fake"}

    monkeypatch.setattr(outlines, "add_outlines", fake_add)
    stats = {}
    _, boxes, _, hits = compose_events.apply_rules(
        {}, cache, rules_path=str(path), video_path="v.mp4", outline_stats=stats,
        log_fn=lambda *a: None)
    assert hits == 0                          # boxes alone would have said yes
    assert stats["frames"] == 3
    assert "contours" not in cache[0]         # the caller's cache is untouched
    assert boxes[0]["contours"]

    _, _, _, hits = compose_events.apply_rules({}, cache, rules_path=str(path),
                                               log_fn=lambda *a: None)
    assert hits == 3                          # no video: decided on boxes, as before


@needs_cv2
def test_the_comparison_tool_draws_both_outliners_and_measures_them(cv2, tmp_path):
    from tools import compare_outliners

    bboxes = [{"timestamp": float(t), "objects": ["thing", "other"],
               "bboxes": [[0.25, 0.25, 0.5, 0.5], [0, 0, 0.1, 0.1]]} for t in range(40)]
    picks = compare_outliners.pick(bboxes, "thing", 8)
    assert len(picks) == 8 and picks[0][0] == 0.0 and picks[-1][0] == 35.0

    square = [[0.3, 0.3], [0.7, 0.3], [0.7, 0.7], [0.3, 0.7]]
    shifted = [[0.4, 0.3], [0.8, 0.3], [0.8, 0.7], [0.4, 0.7]]

    class Fixed:
        def __init__(self, name, outline):
            self.name, self._o = name, outline

        def outline(self, frame, boxes):
            return [self._o for _ in boxes]

    def frames(video, stamps):
        for ts in stamps:
            yield ts, np.zeros((90, 160, 3), np.uint8)

    drawn = {}
    result = compare_outliners.compare(
        "v.mp4", picks[:3], Fixed("one", square), Fixed("two", shifted),
        str(tmp_path / "cmp.jpg"), frame_reader=frames,
        renderer=lambda tiles, caps, cols, path, header="": drawn.update(caps=caps))
    assert result["frames"] == 3 and len(drawn["caps"]) == 3
    assert result["mean_agreement"] == pytest.approx(0.6, abs=0.05)   # 3/5 overlap


# --- SAM through OpenVINO ------------------------------------------------------------------

@needs_cv2
def test_a_frame_is_prepared_as_sams_processor_does(cv2):
    frame = np.full((300, 600, 3), 255, np.uint8)            # wide: width is the long side
    pixels, resized = outlines.sam_preprocess(frame)
    assert pixels.shape == (1, 3, 1024, 1024) and pixels.dtype == np.float32
    assert resized == (512, 1024)
    white = (1.0 - outlines.SAM_MEAN) / outlines.SAM_STD
    assert pixels[0, :, 10, 10] == pytest.approx(white, abs=1e-5)
    assert np.all(pixels[0, :, 600:, :] == 0)               # padding below the picture
    boxes = outlines.sam_boxes([(60, 30, 300, 150)], (300, 600), resized)
    assert boxes.tolist() == [pytest.approx([102.4, 51.2, 512.0, 256.0])]


@needs_cv2
def test_a_low_resolution_mask_comes_back_at_frame_size_without_the_padding(cv2):
    logits = np.full((256, 256), -5.0, np.float32)
    logits[0:64, 0:128] = 5.0                  # a quarter of the height, half the width, of 256
    mask = outlines.sam_mask(logits, (512, 1024), (300, 600))
    assert mask.shape == (300, 600)
    # 64/256 of the 1024 input is 256 rows of the 512 the picture has: half its height.
    ys, xs = np.nonzero(mask)
    assert ys.max() == pytest.approx(149, abs=2) and xs.max() == pytest.approx(299, abs=2)


class _Net:
    """A compiled OpenVINO model stand-in: calls ``fn`` on its inputs."""

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, inputs):
        return self.fn(inputs)


def _box_decoder(inputs):
    """A decoder that answers each box with exactly that box, in mask space."""
    _, boxes = inputs
    masks = np.full((len(boxes), 256, 256), -5.0, np.float32)
    for k, (x1, y1, x2, y2) in enumerate(boxes / 4.0):
        masks[k, int(y1):int(y2), int(x1):int(x2)] = 5.0
    return [masks, np.ones(len(boxes), np.float32)]


def _export(tmp_path, probe_outline):
    folder = tmp_path / "slimsam"
    folder.mkdir()
    for name in (outlines.SAM_ENCODER_FILE, outlines.SAM_DECODER_FILE):
        (folder / name).write_bytes(b"")
    (folder / outlines.SAM_META_FILE).write_text(
        '{"id": "%s", "format": 1, "probe_outline": %s}' % (outlines.SAM_EXPORT_ID, probe_outline),
        encoding="utf-8")
    return str(folder)


def _fake_compile(bad_devices=()):
    compiled = []

    def compile_(path, device):
        compiled.append((os.path.basename(path), device))
        if path.endswith(outlines.SAM_DECODER_FILE):
            return _Net(_box_decoder)
        if device in bad_devices:                            # a GPU computing garbage
            return _Net(lambda inputs: [np.full((1, 256, 64, 64), np.nan, np.float32)])
        return _Net(lambda inputs: [np.zeros((1, 256, 64, 64), np.float32)])
    return compile_, compiled


@needs_cv2
def test_openvino_sam_outlines_each_box_and_checks_its_probe(cv2, tmp_path, monkeypatch):
    frame, box = outlines.sam_probe()
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    probe = [[x1 / w, y1 / h], [x2 / w, y1 / h], [x2 / w, y2 / h], [x1 / w, y2 / h]]
    compile_, compiled = _fake_compile()
    monkeypatch.setattr(outlines.OpenVinoSamOutliner, "_compile", staticmethod(compile_))
    sam = outlines.OpenVinoSamOutliner(folder=_export(tmp_path, probe), device="CPU")
    sam._load()
    assert sam.device == "OpenVINO CPU"
    (decoder, dev), = [c for c in compiled if c[0] == outlines.SAM_DECODER_FILE]
    assert dev == "CPU"                                       # the decoder never goes to the GPU
    image = np.zeros((360, 640, 3), np.uint8)
    got = sam.outline(image, [(64, 36, 320, 180), (500, 300, 501, 301)])
    xs, ys = [p[0] for p in got[0]], [p[1] for p in got[0]]
    assert (min(xs), max(xs)) == (pytest.approx(0.1, abs=0.01), pytest.approx(0.5, abs=0.01))
    assert (min(ys), max(ys)) == (pytest.approx(0.1, abs=0.01), pytest.approx(0.5, abs=0.01))
    assert got[1] is None                                     # nothing usable in a 1-pixel box


@needs_cv2
def test_a_gpu_that_fails_the_probe_is_dropped_for_the_cpu(cv2, tmp_path, monkeypatch):
    frame, box = outlines.sam_probe()
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    probe = [[x1 / w, y1 / h], [x2 / w, y1 / h], [x2 / w, y2 / h], [x1 / w, y2 / h]]

    def nan_decoder_for_nan_embeddings(compile_):
        def wrapped(path, device):
            net = compile_(path, device)
            if path.endswith(outlines.SAM_DECODER_FILE):
                return _Net(lambda inputs: _box_decoder(inputs) if np.isfinite(inputs[0]).all()
                            else [np.full((len(inputs[1]), 256, 256), np.nan, np.float32),
                                  np.zeros(len(inputs[1]), np.float32)])
            return net
        return wrapped

    compile_, compiled = _fake_compile(bad_devices={"GPU"})
    monkeypatch.setattr(outlines.OpenVinoSamOutliner, "_compile",
                        staticmethod(nan_decoder_for_nan_embeddings(compile_)))
    sam = outlines.OpenVinoSamOutliner(folder=_export(tmp_path, probe), device="GPU")
    sam._load()
    assert sam.device == "OpenVINO CPU"
    assert [d for name, d in compiled if name == outlines.SAM_ENCODER_FILE] == ["GPU", "CPU"]


def test_sam_outliner_falls_back_to_transformers(tmp_path, monkeypatch):
    monkeypatch.setattr(outlines, "find_sam_dir", lambda: None)
    assert isinstance(outlines.sam_outliner(), outlines.SamOutliner)

    def broken(self):
        raise RuntimeError("no device")

    monkeypatch.setattr(outlines, "find_sam_dir", lambda: str(tmp_path))
    monkeypatch.setattr(outlines.OpenVinoSamOutliner, "_load", broken)
    assert isinstance(outlines.sam_outliner(), outlines.SamOutliner)
    assert isinstance(outlines.make_outliner("sam"), outlines.SamOutliner)


def test_the_export_is_found_by_override_and_its_meta_is_checked(tmp_path, monkeypatch):
    folder = _export(tmp_path, "[[0, 0], [1, 0], [1, 1]]")
    monkeypatch.setenv(outlines.SAM_DIR_ENV, folder)
    assert outlines.find_sam_dir() == folder
    assert outlines.read_sam_meta(folder)["id"] == outlines.SAM_EXPORT_ID
    meta = os.path.join(folder, outlines.SAM_META_FILE)
    with open(meta, "w", encoding="utf-8") as fh:
        fh.write('{"id": "another-model", "format": 1, "probe_outline": [[0, 0], [1, 0], [1, 1]]}')
    with pytest.raises(ValueError, match="another-model"):
        outlines.read_sam_meta(folder)
    with open(meta, "w", encoding="utf-8") as fh:
        fh.write('{"id": "%s", "format": 99, "probe_outline": [[0, 0], [1, 0], [1, 1]]}'
                 % outlines.SAM_EXPORT_ID)
    with pytest.raises(ValueError, match="newer"):
        outlines.read_sam_meta(folder)


@needs_cv2
def test_outline_iou(cv2):
    square = [[0.1, 0.1], [0.5, 0.1], [0.5, 0.5], [0.1, 0.5]]
    half = [[0.1, 0.1], [0.3, 0.1], [0.3, 0.5], [0.1, 0.5]]
    assert outlines.outline_iou(square, square) == pytest.approx(1.0)
    assert outlines.outline_iou(square, half) == pytest.approx(0.5, abs=0.03)
    assert outlines.outline_iou(square, []) == 0.0
