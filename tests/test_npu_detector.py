"""The NPU detector and the npu_obs tool built on it.

Most of this needs no NPU: the detector is exercised on the CPU device with a
synthetic YOLOX-shaped model whose output is fixed, so a run produces one
known box per frame. Only the parity check at the end needs the real NPU and
the stock models, and skips without them.

conftest.py swaps cv2 and openvino for MagicMocks; tests here that need real
pixels and a real runtime borrow the installed modules through ``real_libs``
and hand the shims back afterwards, as ``real_opencv()`` does for cv2.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import socket
import struct
import sys
import threading
from unittest.mock import MagicMock

import numpy as np
import pytest

from npu_obs.spans import Span, SpanTracker, summarize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- real cv2 / openvino for the tests that need them -----------------------

_REAL: dict = {}


def _heavy(name: str) -> bool:
    return name == "cv2" or name == "openvino" or name.startswith("openvino.")


@pytest.fixture
def real_libs():
    shims = {k: v for k, v in sys.modules.items() if _heavy(k)}
    for k, v in shims.items():
        if isinstance(v, MagicMock):
            del sys.modules[k]
    if _REAL:
        sys.modules.update(_REAL)
    else:
        try:
            import cv2  # noqa: F401
            import openvino  # noqa: F401
            import openvino.preprocess  # noqa: F401
        except Exception:
            for k in [k for k in sys.modules if _heavy(k)]:
                del sys.modules[k]
            sys.modules.update(shims)
            pytest.skip("needs the real OpenCV and OpenVINO")
        _REAL.update({k: v for k, v in sys.modules.items() if _heavy(k)})
    try:
        yield
    finally:
        for k in [k for k in sys.modules if _heavy(k)]:
            del sys.modules[k]
        sys.modules.update(shims)


# --- a YOLOX-shaped model with a known answer -------------------------------

IN = 64                          # input side; strides 8/16/32 -> 64+16+4 rows
ANCHORS = (IN // 8) ** 2 + (IN // 16) ** 2 + (IN // 32) ** 2
NAMES = ["alpha", "beta"]


def _fixed_output(num_classes: int = len(NAMES)) -> np.ndarray:
    """Raw-grid output with one confident box: anchor 0 (grid 0,0 at stride 8)
    decodes to centre (4, 4), size 32x32, class 1 at 0.9 * 1.0."""
    out = np.zeros((1, ANCHORS, 5 + num_classes), np.float32)
    out[0, 0, :4] = [0.5, 0.5, math.log(4.0), math.log(4.0)]
    out[0, 0, 4] = 1.0
    out[0, 0, 5 + 1] = 0.9
    return out


def _make_model(path, out: np.ndarray, in_shape=(1, 3, IN, IN)) -> str:
    """A model whose output is ``out`` whatever the input (the input still
    feeds the graph, so the runtime cannot prune it)."""
    import openvino as ov
    from openvino import opset13 as ops
    x = ops.parameter(ov.PartialShape(list(in_shape)), ov.Type.f32, name="images")
    zero = ops.multiply(ops.reduce_mean(x, ops.constant(np.array([0, 1, 2, 3])), False),
                        ops.constant(np.float32(0)))
    y = ops.add(ops.constant(out), zero)
    model = ov.Model([y], [x], "fixed")
    xml = str(path / "fixed.xml")
    ov.save_model(model, xml, compress_to_fp16=False)
    return xml


def _detector(xml, names=NAMES, **kw):
    from modules.vision.npu_detector import NpuYoloxDetector
    kw.setdefault("device", "CPU")
    kw.setdefault("fallback", ())
    kw.setdefault("cache_dir", "")
    return NpuYoloxDetector(xml, names, **kw)


# --- detector ---------------------------------------------------------------

def test_detects_the_known_box_and_undoes_the_letterbox(real_libs, tmp_path):
    det = _detector(_make_model(tmp_path, _fixed_output()))
    assert det.input_size == (IN, IN)
    frame = np.zeros((128, 256, 3), np.uint8)   # r = 64/256 = 0.25
    (d,) = det.detect(frame)
    assert (d.class_id, d.class_name) == (1, "beta")
    assert d.confidence == pytest.approx(0.9, abs=1e-4)
    # (4,4) +/- 16 in input space, / 0.25 back to the frame
    assert [d.x1, d.y1, d.x2, d.y2] == pytest.approx([-48, -48, 80, 80], abs=1e-3)


def test_uint8_preprocess_matches_the_base_float_path(real_libs, tmp_path):
    from modules.vision.detection_backend import YoloxOpenVINODetector
    xml = _make_model(tmp_path, _fixed_output())
    ours = _detector(xml)
    base = YoloxOpenVINODetector(xml, NAMES, device="CPU")
    frame = np.random.default_rng(1).integers(0, 256, (90, 160, 3), dtype=np.uint8)
    a, ra = ours._preprocess(frame)
    b, rb = base._preprocess(frame)
    assert ra == rb
    assert a.dtype == np.uint8 and a.shape == (1, IN, IN, 3)
    np.testing.assert_array_equal(a.transpose(0, 3, 1, 2).astype(np.float32), b)


def test_unknown_device_falls_back_and_says_so(real_libs, tmp_path):
    said = []
    det = _detector(_make_model(tmp_path, _fixed_output()),
                    device="NO_SUCH_DEVICE", fallback=("CPU",), log=said.append)
    assert det.device == "CPU"
    assert said and "NO_SUCH_DEVICE" in said[0] and "CPU" in said[0]


def test_no_fallback_means_an_error_not_a_quiet_cpu_run(real_libs, tmp_path):
    with pytest.raises(RuntimeError, match="NO_SUCH_DEVICE"):
        _detector(_make_model(tmp_path, _fixed_output()), device="NO_SUCH_DEVICE")


def test_dynamic_input_is_pinned_static(real_libs, tmp_path):
    xml = _make_model(tmp_path, _fixed_output(), in_shape=(-1, 3, IN, IN))
    det = _detector(xml)
    assert det._compiled.input(0).partial_shape.is_static
    assert len(det.detect(np.zeros((64, 64, 3), np.uint8))) == 1


def test_refuses_channels_first_layout(real_libs, tmp_path):
    out = np.zeros((1, 4 + len(NAMES), ANCHORS), np.float32)
    with pytest.raises(ValueError, match="channels-first"):
        _detector(_make_model(tmp_path, out))


def test_refuses_a_class_count_the_names_do_not_match(real_libs, tmp_path):
    with pytest.raises(ValueError, match="3 names"):
        _detector(_make_model(tmp_path, _fixed_output()), names=["a", "b", "c"])


def test_refuses_the_wrong_number_of_rows(real_libs, tmp_path):
    out = np.zeros((1, ANCHORS - 4, 5 + len(NAMES)), np.float32)
    with pytest.raises(ValueError, match="rows"):
        _detector(_make_model(tmp_path, out))


def test_build_reads_labels_json_beside_a_custom_model(real_libs, tmp_path):
    from modules.vision.npu_detector import build_npu_detector
    xml = _make_model(tmp_path, _fixed_output())
    (tmp_path / "labels.json").write_text(json.dumps(NAMES), encoding="utf-8")
    det = build_npu_detector(xml, device="CPU", fallback=(), cache_dir="")
    assert det.class_names == NAMES


def test_build_without_names_is_an_error(real_libs, tmp_path):
    from modules.vision.npu_detector import build_npu_detector
    with pytest.raises(ValueError, match="No class names"):
        build_npu_detector(_make_model(tmp_path, _fixed_output()), device="CPU",
                           fallback=(), cache_dir="")


# --- spans ------------------------------------------------------------------

def _feed(tracker, samples):
    events = []
    for t, seen in samples:
        events += [(k, s.label, s.start, s.end) for k, s in tracker.update(t, seen)]
    events += [(k, s.label, s.start, s.end) for k, s in tracker.finish()]
    return events


def test_span_needs_min_hits_in_a_row():
    tr = SpanTracker(min_hits=2, max_gap=1.0)
    events = _feed(tr, [(0.0, {"a": 0.5}), (0.5, {}), (1.0, {"a": 0.5}), (1.5, {"a": 0.6})])
    assert events[0] == ("start", "a", 0.0, 1.5)


def test_one_sample_blip_never_opens_a_span():
    tr = SpanTracker(min_hits=2, max_gap=1.0)
    assert _feed(tr, [(0.0, {"a": 0.9}), (0.5, {}), (3.0, {})]) == []
    assert tr.closed == []


def test_a_short_dropout_does_not_split_the_span():
    tr = SpanTracker(min_hits=2, max_gap=1.0)
    samples = [(0.0, {"a": .5}), (0.5, {"a": .5}), (1.0, {}), (1.5, {"a": .7}),
               (2.0, {"a": .5}), (4.0, {})]
    events = _feed(tr, samples)
    assert [e[0] for e in events] == ["start", "end"]
    (span,) = tr.closed
    assert (span.start, span.end, span.peak, span.hits) == (0.0, 2.0, .7, 4)


def test_a_long_gap_ends_the_span_and_a_return_starts_another():
    tr = SpanTracker(min_hits=1, max_gap=1.0)
    _feed(tr, [(0.0, {"a": .5}), (5.0, {"a": .5})])
    assert [(s.start, s.end) for s in tr.closed] == [(0.0, 0.0), (5.0, 5.0)]


def test_summary_totals_per_label_longest_first():
    spans = [Span("a", 0, 2, .5, 3), Span("b", 0, 10, .9, 9), Span("a", 5, 6, .8, 2)]
    s = summarize(spans)
    assert list(s) == ["b", "a"]
    assert s["a"] == {"spans": 2, "seconds": 3.0, "peak": 0.8}


# --- video source + CLI end to end -------------------------------------------

def _write_video(path, seconds=4, fps=10, size=(96, 64)):
    import cv2
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(seconds * fps):
        vw.write(np.full((size[1], size[0], 3), i % 255, np.uint8))
    vw.release()
    return str(path)


def test_video_source_samples_on_schedule(real_libs, tmp_path):
    from npu_obs.sources import VideoFileSource, prefetch
    src = VideoFileSource(_write_video(tmp_path / "v.mp4"), every=0.5, hw_decode=False)
    times = [round(t, 3) for t, _ in prefetch(src)]
    src.close()
    assert times == [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    assert src.samples == len(times)


def test_video_source_honours_start_and_end(real_libs, tmp_path):
    from npu_obs.sources import VideoFileSource
    src = VideoFileSource(_write_video(tmp_path / "v.mp4"), every=0.5,
                          hw_decode=False, start=1.0, end=2.6)
    times = [round(t, 3) for t, _ in src]
    src.close()
    assert times == [1.0, 1.5, 2.0, 2.5]


def test_closing_prefetch_stops_its_worker():
    import time as _time

    from npu_obs.sources import prefetch
    produced = []

    def endless():
        while True:
            produced.append(len(produced))
            yield produced[-1]

    gen = prefetch(endless(), depth=2)
    assert [next(gen), next(gen)] == [0, 1]
    gen.close()
    settled = len(produced)
    _time.sleep(0.5)
    assert len(produced) == settled <= 6


def test_capture_source_hands_out_the_last_frame_then_stops(real_libs, tmp_path):
    from npu_obs.sources import CaptureSource
    src = CaptureSource(_write_video(tmp_path / "v.mp4", seconds=1))
    try:
        _, frame = src.read()
        assert frame.shape == (64, 96, 3)
        with pytest.raises(RuntimeError, match="stopped"):
            for _ in range(100):   # a file runs dry; a stream would too
                src.read()
    finally:
        src.close()


def test_video_source_uses_frame_timestamps_not_frame_counts(real_libs, tmp_path):
    """OBS records at a variable frame rate. Here: 10 fps for 2 s, then 5 fps.
    Counting frames at the 6.9 fps average would put every later sample at
    the wrong time; the frames' own timestamps do not drift."""
    import shutil
    import subprocess
    from npu_obs.sources import VideoFileSource
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("needs ffmpeg to build a variable-frame-rate clip")
    path = str(tmp_path / "vfr.mp4")
    subprocess.run([ffmpeg, "-v", "error", "-y", "-f", "lavfi",
                    "-i", "testsrc=size=96x64:rate=10:duration=4",
                    "-vf", "setpts='if(lt(N,20),N/10,2+(N-20)/5)/TB'",
                    "-fps_mode", "vfr", "-c:v", "mpeg4", path], check=True)
    src = VideoFileSource(path, every=0.5, hw_decode=False)
    times = [round(t, 2) for t, _ in src]
    src.close()
    assert times[:5] == [0.0, 0.5, 1.0, 1.5, 2.0]
    # 5 fps from 2.0 s: each sample is the first frame at or after its slot
    assert all(a < b for a, b in zip(times, times[1:]))
    assert times[-1] == pytest.approx(5.5, abs=0.25)
    for t, slot in zip(times, [i * 0.5 for i in range(len(times))]):
        assert slot - 1e-6 <= t < slot + 0.25


def test_prefetch_passes_errors_through():
    from npu_obs.sources import prefetch

    def items():
        yield 1
        raise OSError("disk gone")

    got = []
    with pytest.raises(OSError, match="disk gone"):
        for x in prefetch(items()):
            got.append(x)
    assert got == [1]


def test_cli_video_writes_one_span_for_a_box_in_every_frame(real_libs, tmp_path):
    from npu_obs.cli import main
    xml = _make_model(tmp_path, _fixed_output())
    (tmp_path / "labels.json").write_text(json.dumps(NAMES), encoding="utf-8")
    video = _write_video(tmp_path / "v.mp4")
    out = tmp_path / "r.json"
    rc = main(["video", video, "--model", xml, "--device", "CPU", "--npu-only",
               "--every", "0.5", "--cpu-decode", "--normal-priority", "--out", str(out)])
    assert rc == 0
    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["detector"]["device"] == "CPU"
    assert result["sampling"]["samples"] == 8 and result["sampling"]["complete"]
    assert result["spans"] == [{"label": "beta", "start": 0.0, "end": 3.5,
                                "peak": pytest.approx(0.9, abs=1e-3), "hits": 8}]


def test_cli_rejects_a_class_the_model_does_not_have(real_libs, tmp_path):
    from npu_obs.cli import main
    xml = _make_model(tmp_path, _fixed_output())
    (tmp_path / "labels.json").write_text(json.dumps(NAMES), encoding="utf-8")
    with pytest.raises(SystemExit, match="gamma"):
        main(["video", _write_video(tmp_path / "v.mp4"), "--model", xml,
              "--device", "CPU", "--classes", "beta,gamma", "--normal-priority"])


# --- obs-websocket client against a fake OBS -----------------------------------

_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


class FakeObs:
    """Just enough obs-websocket v5 to answer the client: a real TCP server
    speaking WebSocket framing, so the client's own stack is what's tested."""

    def __init__(self, password: str = "", record_ms: int | None = 61_500):
        self.password = password
        self.record_ms = record_ms
        self.scene = "Game scene"
        self.chapters: list[str] = []
        self.requests: list[str] = []
        self._srv = socket.socket()
        self._srv.bind(("127.0.0.1", 0))
        self._srv.listen(4)
        self.port = self._srv.getsockname()[1]
        threading.Thread(target=self._serve, daemon=True).start()

    def close(self):
        self._srv.close()

    # wire
    def _serve(self):
        while True:
            try:
                conn, _ = self._srv.accept()
            except OSError:
                return
            threading.Thread(target=self._session, args=(conn,), daemon=True).start()

    @staticmethod
    def _recv_exact(conn, n):
        buf = b""
        while len(buf) < n:
            chunk = conn.recv(n - len(buf))
            if not chunk:
                raise ConnectionError
            buf += chunk
        return buf

    def _recv(self, conn):
        b0, b1 = self._recv_exact(conn, 2)
        n = b1 & 0x7F
        if n == 126:
            n = struct.unpack(">H", self._recv_exact(conn, 2))[0]
        elif n == 127:
            n = struct.unpack(">Q", self._recv_exact(conn, 8))[0]
        mask = self._recv_exact(conn, 4) if b1 & 0x80 else b"\0\0\0\0"
        data = bytes(c ^ mask[i % 4] for i, c in enumerate(self._recv_exact(conn, n)))
        if b0 & 0x0F == 8:
            raise ConnectionError
        return json.loads(data)

    @staticmethod
    def _send(conn, msg, opcode=1):
        data = msg if isinstance(msg, bytes) else json.dumps(msg).encode()
        head = bytes([0x80 | opcode])
        if len(data) < 126:
            head += bytes([len(data)])
        elif len(data) < 65536:
            head += bytes([126]) + struct.pack(">H", len(data))
        else:
            head += bytes([127]) + struct.pack(">Q", len(data))
        conn.sendall(head + data)

    def _session(self, conn):
        try:
            req = b""
            while b"\r\n\r\n" not in req:
                req += conn.recv(4096)
            headers = dict(line.split(": ", 1) for line in
                           req.decode().split("\r\n")[1:] if ": " in line)
            key = headers["Sec-WebSocket-Key"]
            accept = base64.b64encode(hashlib.sha1((key + _WS_GUID).encode()).digest()).decode()
            conn.sendall(("HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\n"
                          "Connection: Upgrade\r\nSec-WebSocket-Accept: " + accept +
                          "\r\nSec-WebSocket-Protocol: obswebsocket.json\r\n\r\n").encode())
            hello = {"obsWebSocketVersion": "5.5.0", "rpcVersion": 1}
            salt, challenge = "c2FsdA==", "Y2hhbGxlbmdl"
            if self.password:
                hello["authentication"] = {"salt": salt, "challenge": challenge}
            self._send(conn, {"op": 0, "d": hello})
            ident = self._recv(conn)
            if self.password:
                secret = base64.b64encode(hashlib.sha256(
                    (self.password + salt).encode()).digest()).decode()
                want = base64.b64encode(hashlib.sha256(
                    (secret + challenge).encode()).digest()).decode()
                if ident["d"].get("authentication") != want:
                    self._send(conn, struct.pack(">H", 4009) + b"auth failed", opcode=8)
                    return
            self._send(conn, {"op": 2, "d": {"negotiatedRpcVersion": 1}})
            while True:
                msg = self._recv(conn)
                d = msg["d"]
                self.requests.append(d["requestType"])
                ok, data = self._answer(d["requestType"], d.get("requestData") or {})
                self._send(conn, {"op": 7, "d": {
                    "requestType": d["requestType"], "requestId": d["requestId"],
                    "requestStatus": {"result": ok, "code": 100 if ok else 600,
                                      **({} if ok else {"comment": "nope"})},
                    **({"responseData": data} if data else {})}})
        except (ConnectionError, OSError, KeyError):
            pass
        finally:
            conn.close()

    def _answer(self, kind, data):
        import cv2
        if kind == "GetVersion":
            return True, {"obsVersion": "31.0.0"}
        if kind == "GetCurrentProgramScene":
            return True, {"sceneName": self.scene, "currentProgramSceneName": self.scene}
        if kind == "GetSourceScreenshot":
            if data.get("sourceName") != self.scene:
                return False, None
            w = int(data["imageWidth"])
            img = np.full((w * 9 // 16, w, 3), 80, np.uint8)
            ok, jpg = cv2.imencode(".jpg", img)
            return True, {"imageData": "data:image/jpg;base64," +
                          base64.b64encode(jpg.tobytes()).decode()}
        if kind == "GetRecordStatus":
            active = self.record_ms is not None
            return True, {"outputActive": active, "outputPaused": False,
                          "outputDuration": self.record_ms or 0}
        if kind == "CreateRecordChapter":
            self.chapters.append(data["chapterName"])
            return True, None
        return False, None


@pytest.fixture
def fake_obs():
    pytest.importorskip("websocket")
    servers = []

    def make(**kw):
        s = FakeObs(**kw)
        servers.append(s)
        return s

    yield make
    for s in servers:
        s.close()


def test_auth_string_follows_the_protocol():
    from npu_obs.obs_ws import auth_string
    secret = base64.b64encode(hashlib.sha256(b"pwsalt").digest()).decode()
    want = base64.b64encode(hashlib.sha256((secret + "chal").encode()).digest()).decode()
    assert auth_string("pw", "salt", "chal") == want


def test_client_authenticates_and_requests(fake_obs):
    from npu_obs.obs_ws import ObsClient
    obs = fake_obs(password="hunter2")
    with ObsClient("127.0.0.1", obs.port, "hunter2") as c:
        assert c.request("GetVersion")["obsVersion"] == "31.0.0"
        assert c.program_scene() == "Game scene"
        assert c.record_status()["outputDuration"] == 61_500


def test_wrong_password_is_reported_as_such(fake_obs):
    from npu_obs.obs_ws import ObsClient, ObsError
    obs = fake_obs(password="hunter2")
    with pytest.raises(ObsError, match="password"):
        ObsClient("127.0.0.1", obs.port, "wrong").connect()


def test_missing_password_is_reported_before_trying(fake_obs):
    from npu_obs.obs_ws import ObsClient, ObsError
    obs = fake_obs(password="hunter2")
    with pytest.raises(ObsError, match="OBS_WEBSOCKET_PASSWORD"):
        ObsClient("127.0.0.1", obs.port, "").connect()


def test_failed_request_raises_with_obs_comment(fake_obs):
    from npu_obs.obs_ws import ObsClient, ObsError
    obs = fake_obs()
    with ObsClient("127.0.0.1", obs.port) as c, pytest.raises(ObsError, match="nope"):
        c.request("NoSuchRequest")


def test_nothing_listening_says_how_to_turn_obs_websocket_on():
    pytest.importorskip("websocket")
    from npu_obs.obs_ws import ObsClient, ObsError
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    with pytest.raises(ObsError, match="WebSocket server"):
        ObsClient("127.0.0.1", port, timeout=1).connect()


def test_screenshot_source_follows_the_program_scene(real_libs, fake_obs):
    from npu_obs.obs_ws import ObsClient
    from npu_obs.sources import ObsScreenshotSource
    obs = fake_obs()
    with ObsClient("127.0.0.1", obs.port) as c:
        src = ObsScreenshotSource(c, width=320)
        _, frame = src.read()
        assert frame.shape == (180, 320, 3)
        obs.scene = "Other scene"           # user switched scenes
        _, frame = src.read()
        assert src.source == "Other scene" and frame.shape == (180, 320, 3)
        assert src.record_ms() == 61_500
        obs.record_ms = None
        assert src.record_ms() is None


def test_cli_live_logs_spans_and_adds_chapters(real_libs, fake_obs, tmp_path, monkeypatch):
    import time as _time

    from npu_obs import cli
    obs = fake_obs()
    xml = _make_model(tmp_path, _fixed_output())
    (tmp_path / "labels.json").write_text(json.dumps(NAMES), encoding="utf-8")
    out = tmp_path / "live.jsonl"

    # Stop the otherwise endless loop after a few samples, as Ctrl+C would.
    calls = {"n": 0}
    real_sleep = _time.sleep

    def sleep(s):
        calls["n"] += 1
        if calls["n"] >= 4:
            raise KeyboardInterrupt
        real_sleep(min(s, 0.01))

    monkeypatch.setattr(cli.time, "sleep", sleep)
    rc = cli.main(["live", "--obs-host", "127.0.0.1", "--obs-port", str(obs.port),
                   "--model", xml, "--device", "CPU", "--npu-only", "--interval", "0.05",
                   "--chapters", "--normal-priority", "--out", str(out)])
    assert rc == 0
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [r["event"] for r in rows] == ["start", "end"]
    assert rows[0]["label"] == "beta"
    # the fake recording stands still at 61.5 s, so positions sit just before it
    assert all(r["record_ms"] is not None and r["record_ms"] <= 61_500 for r in rows)
    assert obs.chapters == ["beta"]


# --- the real thing ---------------------------------------------------------

def test_npu_matches_cpu_on_the_stock_model(real_libs):
    from modules.vision.detection_backend import find_default_yolox_ir, load_class_names
    from modules.vision.npu_detector import NpuYoloxDetector, npu_device
    xml = find_default_yolox_ir(prefer="small")
    if not npu_device() or not xml:
        pytest.skip("needs an Intel NPU and an installed YOLOX model")
    names = load_class_names(os.path.join(ROOT, "yolo_objects_labels.json"))
    frame = np.random.default_rng(3).integers(0, 256, (360, 640, 3), dtype=np.uint8)
    npu = NpuYoloxDetector(xml, names, device="NPU", fallback=(), cache_dir="")
    cpu = NpuYoloxDetector(xml, names, device="CPU", fallback=(), cache_dir="")
    assert npu.device == "NPU"
    blob, _ = npu._preprocess(frame)
    a = np.array(npu._infer(blob))[0]
    b = np.array(cpu._infer(blob))[0]
    # FP16 on the NPU. Compare the scores (objectness x class, what thresholds
    # see); raw box columns are log-space sizes, where FP16 drift on anchors
    # nobody keeps looks large and means nothing.
    score_a, score_b = a[:, 4:5] * a[:, 5:], b[:, 4:5] * b[:, 5:]
    assert np.abs(score_a - score_b).max() < 0.02
