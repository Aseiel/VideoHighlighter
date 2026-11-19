"""Views: what of each frame a head sees, the same in training and in use.

Measured 2026-11-19: whole frames 52.8 % top-1 on held-out videos, whole
frames + quarters + centre 57.2 %, higher in every seed. These tests pin what
makes that number reproducible in the app: one definition of the views, in one
order, recorded in head.json and read back wherever the head is used.
"""
from __future__ import annotations

import json
import types

import numpy as np
import pytest

from model_training.action_head import features as Fx
from model_training.action_head import train as T
from model_training.action_head import views as V


def frame(value, h=8, w=12):
    return np.full((h, w, 3), value, np.uint8)


def test_counts():
    assert V.per_frame(V.WHOLE) == 1 and V.per_frame(V.TILES) == 6
    assert V.tokens(4, V.WHOLE) == 4 and V.tokens(4, V.TILES) == 24
    with pytest.raises(ValueError, match="views must be"):
        V.tokens(4, "sideways")


def test_whole_is_the_frames_unchanged():
    frames = [frame(1), frame(2)]
    assert V.expand(frames, V.WHOLE) == frames


def test_tiles_follow_the_whole_frames_in_frame_order():
    f = np.zeros((8, 12, 3), np.uint8)
    f[:4, :6], f[:4, 6:], f[4:, :6], f[4:, 6:] = 1, 2, 3, 4
    out = V.expand([f, f + 10], V.TILES)
    assert len(out) == 12
    assert out[0] is not None and out[1] is not None          # the whole frames first
    np.testing.assert_array_equal(out[0], f)
    tiles_first = [int(t[0, 0, 0]) for t in out[2:6]]          # quarters of frame 1
    assert tiles_first == [1, 2, 3, 4]
    centre = out[6]
    assert centre.shape[:2] == (4, 6)                          # the middle half each way
    assert [int(t[0, 0, 0]) for t in out[7:11]] == [11, 12, 13, 14]   # frame 2's quarters


def test_a_head_without_the_field_is_whole():
    assert V.of({}) == V.WHOLE and V.of({"views": V.TILES}) == V.TILES
    assert V.of(types.SimpleNamespace()) == V.WHOLE
    assert V.of(types.SimpleNamespace(views=V.TILES)) == V.TILES


class CountingEncoder:
    encoder_id, label, dims = "enc-a", "test", 8

    def __init__(self):
        self.images = []

    def encode_bgr(self, images):
        self.images.append(len(images))
        return np.arange(len(images) * self.dims, dtype=np.float32).reshape(len(images), self.dims)


def test_encoding_with_tiles_reads_frames_and_stores_every_view(monkeypatch, tmp_path):
    clip = tmp_path / "a.mp4"
    clip.write_bytes(b"a")
    asked = []
    monkeypatch.setattr(Fx, "read_frames", lambda p, k: asked.append(k) or [frame(k)] * k)
    cache = Fx.FeatureCache(str(tmp_path / "c.npz"), "enc-a", V.tokens(4, V.TILES), 8)
    enc = CountingEncoder()
    feats, ok = Fx.encode_clips([str(clip)], str(tmp_path), enc, cache, log=lambda m: None,
                                views=V.TILES)
    assert asked == [4]                    # 4 frames read from the clip
    assert enc.images == [24]              # 24 images encoded
    assert feats.shape == (1, 24, 8) and ok.all()


def test_a_cache_that_does_not_fit_the_views_is_refused(tmp_path):
    cache = Fx.FeatureCache(str(tmp_path / "c.npz"), "enc-a", 4, 8)
    with pytest.raises(ValueError, match="does not fit"):
        Fx.encode_clips([], str(tmp_path), CountingEncoder(), cache, views=V.TILES)


def test_whole_frame_caches_keep_their_name(monkeypatch, tmp_path):
    from modules.system import app_paths
    monkeypatch.setattr(app_paths, "user_data_dir", lambda: str(tmp_path))
    whole = T.default_cache("D:/data", "enc", 4)
    assert whole == T.default_cache("D:/data", "enc", 4, V.WHOLE)
    assert whole.endswith("-enc-4f.npz")
    assert T.default_cache("D:/data", "enc", 4, V.TILES).endswith("-enc-4f-whole-tiles.npz")


def test_the_trainer_takes_views_and_refuses_them_with_fine_tuning():
    logs = []
    code = T.main(["--data-path", "nowhere", "--views", V.TILES, "--finetune-blocks", "1"],
                  log=logs.append)
    assert code == 1 and any("whole frames only" in m for m in logs)


def test_the_sorter_expands_frames_as_the_head_was_trained():
    from modules.teach import sort

    seen = []

    class Head:
        frames, views, classes, encoder_id = 2, V.TILES, ["a"], None

        def scores(self, feats):
            seen.append(feats.shape)
            return np.zeros((1, 1))

    class Enc:
        encoder_id = None

        def encode_bgr(self, images):
            return np.zeros((len(images), 8), np.float32)

    score, classes = sort._head_model("x", encoder=Enc(), head=Head(),
                                      frame_reader=lambda p, k: [frame(1)] * k)
    score("clip.mp4")
    assert seen == [(1, 12, 8)]            # 2 frames x 6 views


def test_the_app_reads_the_views_a_head_was_trained_with(tmp_path, monkeypatch):
    """head.json's views reach the loaded head; a head without them is whole.
    (That a head takes any number of vectors per clip is test_action_head's.)"""
    import sys
    from modules.vision import action_siglip as A

    class Session:
        def __init__(self, *a, **k):
            pass

        def get_inputs(self):
            return [types.SimpleNamespace(name="features")]

        def run(self, _, feeds):
            return [np.zeros((len(feeds["features"]), 2), np.float32)]

    monkeypatch.setitem(sys.modules, "onnxruntime",
                        types.SimpleNamespace(InferenceSession=Session))
    for views, folder in ((V.TILES, tmp_path / "tiles"), (None, tmp_path / "old")):
        folder.mkdir()
        (folder / A.HEAD_MODEL).write_bytes(b"")
        meta = {"kind": A.HEAD_KIND, "encoder": "enc-a", "frames": 4,
                "classes": ["a", "b"], "trust_thresholds": [0.5, 0.5]}
        if views:
            meta["views"] = views
        (folder / A.HEAD_META).write_text(json.dumps(meta))
    tiles, old = A.ActionHead(str(tmp_path / "tiles")), A.ActionHead(str(tmp_path / "old"))
    assert tiles.views == V.TILES and old.views == V.WHOLE
    assert tiles.scores(np.zeros((3, 24, 8), np.float32)).shape == (3, 2)
