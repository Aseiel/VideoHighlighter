"""Clips are read a few ahead while the encoder works, and nothing else changes.

Measured 2026-11-20, 200 dataset clips, Arc A750: whole frames 12.0 -> 19.8
clips/s, whole frames + tiles 7.6 -> 10.1, vectors identical to the last bit.
"""
from __future__ import annotations

import threading
import time

import numpy as np

from model_training.action_head import features as Fx


def test_results_come_back_in_order_even_when_reads_finish_out_of_order():
    def slow_first(n):
        time.sleep(0.05 if n % 3 == 0 else 0)
        return n * 10

    got = list(Fx.prefetch(range(20), slow_first, ahead=4, threads=4))
    assert got == [(n, n * 10) for n in range(20)]


def test_it_reads_ahead_but_no_further_than_asked():
    started, gate = [], threading.Event()

    def read(n):
        started.append(n)
        if n == 0:
            gate.wait(2)
        return n

    gen = Fx.prefetch(range(50), read, ahead=3, threads=8)
    t = threading.Thread(target=lambda: next(gen))
    t.start()
    time.sleep(0.2)
    assert sorted(started) == [0, 1, 2, 3]       # the one handed out + 3 ahead
    gate.set()
    t.join(2)
    gen.close()


def test_stopping_early_does_not_read_the_rest():
    started = []

    def read(n):
        started.append(n)
        return n

    for item, _ in Fx.prefetch(range(100), read, ahead=2, threads=1):
        if item == 5:
            break
    assert max(started) <= 8


def test_encoding_with_prefetch_gives_each_clip_its_own_frames(monkeypatch, tmp_path):
    clips = []
    for i in range(12):
        p = tmp_path / f"c{i}.mp4"
        p.write_bytes(bytes([i]) * (i + 1))
        clips.append(str(p))

    def read_frames(path, k):
        time.sleep(0.02 if path.endswith(("c0.mp4", "c5.mp4")) else 0)
        v = int(path.rsplit("c", 1)[1].split(".")[0])
        return [np.full((4, 4, 3), v, np.uint8)] * k

    class Enc:
        encoder_id, label, dims = "enc", "test", 2

        def encode_bgr(self, images):
            return np.array([[im[0, 0, 0], 1] for im in images], np.float32)

    monkeypatch.setattr(Fx, "read_frames", read_frames)
    cache = Fx.FeatureCache(str(tmp_path / "c.npz"), "enc", 4, 2)
    feats, ok = Fx.encode_clips(clips, str(tmp_path), Enc(), cache, log=lambda m: None)
    assert ok.all()
    assert [int(f[0, 0]) for f in feats] == list(range(12))
