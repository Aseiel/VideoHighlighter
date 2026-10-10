"""
Per-frame box tracking between live detections.

The live detectors are slow next to the video: faces are detected about twice
a second, and a box used to sit where the last detection put it until the next
one arrived — measured 230 ms behind the picture at the median and ~870 ms at
p95 on a moving subject. This follows every box on every displayed frame and
uses the detector only to correct, start and end tracks.

Per track, on each frame:

  1. Predict   constant-velocity Kalman filter on the box centre.
  2. Match     MOSSE correlation filter (Bolme et al. 2010) over a window of
               2x the box around the prediction; the peak is the measurement
               and its peak-to-sidelobe ratio (PSR) is the confidence.
  3. Gate      a peak further than _MAX_STEP box-sizes from the prediction is
               not believed, however sharp.
  4. FSM       TRACKING   confident: take the measurement, adapt appearance
               UNCERTAIN  middling: take the measurement, freeze appearance
                          (adapting on a doubtful match is how trackers drift
                          onto the background)
               LOST       no believable match: coast on the Kalman prediction
                          for a moment, then stop drawing the box
               A detection re-anchors a track in any state.

Detections describe an older frame than the one on screen — the detector was
handed it inference-time ago. So a detection is compared with where the track
was *at that frame* (each track keeps a short position history) and only the
difference is applied to where it is now. Snapping the box to the detection
instead would yank it back to a stale position on every correction.

The tracker sees only the luma plane, which the display path already holds at
screen size (display_frames.py), so it costs no colour conversion. Pure numpy,
no Qt: LiveBoxTracker is the model, LiveTrackController feeds it frames.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Optional

import numpy as np

TRACKING, UNCERTAIN, LOST = "tracking", "uncertain", "lost"


# ──────────────────────────────────────────────────────────────────
# MOSSE correlation filter
# ──────────────────────────────────────────────────────────────────

_PATCH = 64            # filter size; the box's 2x context window is resampled to this
_PADDING = 2.0         # context window = box size * padding
_LEARN_RATE = 0.125    # appearance adaptation per confident frame (Bolme's value)
_SIGMA = 2.0           # target Gaussian, in patch pixels
_EPS = 1e-2            # regulariser for the filter's denominator


def _hann(n: int) -> np.ndarray:
    w = np.hanning(n).astype(np.float32)
    return np.outer(w, w)


_WINDOW = _hann(_PATCH)


def _target_fft() -> np.ndarray:
    ax = np.arange(_PATCH, dtype=np.float32) - _PATCH // 2
    g = np.exp(-(ax[None, :] ** 2 + ax[:, None] ** 2) / (2 * _SIGMA ** 2))
    return np.fft.fft2(np.fft.ifftshift(g))


_G = _target_fft()


def sample_patch(gray: np.ndarray, cx: float, cy: float, w: float, h: float) -> np.ndarray:
    """Bilinear-resample the (w x h) window centred on (cx, cy) to _PATCH x _PATCH,
    clamping at the frame edge."""
    H, W = gray.shape
    t = (np.arange(_PATCH, dtype=np.float32) + 0.5) / _PATCH - 0.5
    xs = np.clip(cx + t * w, 0, W - 1.001)
    ys = np.clip(cy + t * h, 0, H - 1.001)
    x0 = xs.astype(np.int32)
    y0 = ys.astype(np.int32)
    fx = (xs - x0)[None, :]
    fy = (ys - y0)[:, None]
    g = gray.astype(np.float32, copy=False)
    a = g[y0[:, None], x0[None, :]]
    b = g[y0[:, None], x0[None, :] + 1]
    c = g[y0[:, None] + 1, x0[None, :]]
    d = g[y0[:, None] + 1, x0[None, :] + 1]
    top = a + (b - a) * fx
    bot = c + (d - c) * fx
    return top + (bot - top) * fy


def _preprocess(patch: np.ndarray) -> np.ndarray:
    p = np.log1p(patch)
    p = (p - p.mean()) / (p.std() + 1e-5)
    return p * _WINDOW


class Mosse:
    def __init__(self, patch: np.ndarray):
        F = np.fft.fft2(_preprocess(patch))
        # Bolme initialises from several perturbed copies of the first patch;
        # small shifts are enough to keep the first few frames from overfitting.
        self._A = _G * np.conj(F)
        self._B = F * np.conj(F)
        for dx, dy in ((2, 0), (-2, 0), (0, 2), (0, -2)):
            Fs = np.fft.fft2(_preprocess(np.roll(patch, (dy, dx), axis=(0, 1))))
            Gs = _G * np.exp(-2j * np.pi * (np.fft.fftfreq(_PATCH)[None, :] * dx
                                            + np.fft.fftfreq(_PATCH)[:, None] * dy))
            self._A += Gs * np.conj(Fs)
            self._B += Fs * np.conj(Fs)

    def respond(self, patch: np.ndarray) -> tuple[float, float, float]:
        """(dx, dy) of the response peak from the patch centre, in patch pixels,
        and the peak-to-sidelobe ratio."""
        H = self._A / (self._B + _EPS)
        r = np.real(np.fft.ifft2(H * np.fft.fft2(_preprocess(patch))))
        r = np.fft.fftshift(r)
        iy, ix = np.unravel_index(int(np.argmax(r)), r.shape)
        peak = r[iy, ix]
        mask = np.ones_like(r, dtype=bool)
        mask[max(0, iy - 5):iy + 6, max(0, ix - 5):ix + 6] = False
        side = r[mask]
        psr = float((peak - side.mean()) / (side.std() + 1e-6))
        return float(ix - _PATCH // 2), float(iy - _PATCH // 2), psr

    def adapt(self, patch: np.ndarray, rate: Optional[float] = None) -> None:
        rate = _LEARN_RATE if rate is None else rate
        F = np.fft.fft2(_preprocess(patch))
        self._A = rate * (_G * np.conj(F)) + (1 - rate) * self._A
        self._B = rate * (F * np.conj(F)) + (1 - rate) * self._B


# ──────────────────────────────────────────────────────────────────
# Kalman filter on the box centre: state [cx, cy, vx, vy], per-second velocity
# ──────────────────────────────────────────────────────────────────

class _Kalman:
    _Q = 400.0     # process noise (px/s^2)^2 scale — subjects accelerate
    _R = 4.0       # measurement noise (px^2) — MOSSE peaks are sharp

    def __init__(self, cx: float, cy: float):
        self.x = np.array([cx, cy, 0.0, 0.0])
        self.P = np.diag([10.0, 10.0, 1e4, 1e4])

    def predict(self, dt: float) -> None:
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        q = self._Q
        Q = np.array([[dt ** 4 / 4, 0, dt ** 3 / 2, 0],
                      [0, dt ** 4 / 4, 0, dt ** 3 / 2],
                      [dt ** 3 / 2, 0, dt ** 2, 0],
                      [0, dt ** 3 / 2, 0, dt ** 2]]) * q
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, cx: float, cy: float) -> None:
        Hm = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
        y = np.array([cx, cy]) - Hm @ self.x
        S = Hm @ self.P @ Hm.T + np.eye(2) * self._R
        K = self.P @ Hm.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ Hm) @ self.P

    def shift(self, dx: float, dy: float) -> None:
        self.x[0] += dx
        self.x[1] += dy


# ──────────────────────────────────────────────────────────────────
# Tracks + the tracker
# ──────────────────────────────────────────────────────────────────

class _Track:
    def __init__(self, tid: int, gray, box, payload: dict, t: float):
        x1, y1, x2, y2 = box
        self.id = tid
        self.w, self.h = max(4.0, x2 - x1), max(4.0, y2 - y1)
        self.kf = _Kalman((x1 + x2) / 2, (y1 + y2) / 2)
        self.state = TRACKING
        self.psr = 0.0
        self.lost_since: Optional[float] = None
        self.misses = 0            # detection rounds in a row that didn't see it
        self.payload = payload
        self.history: deque = deque(maxlen=120)   # (t, cx, cy): ~4s at 30fps
        self.history.append((t, self.cx, self.cy))
        self.filter = Mosse(self._patch(gray))

    @property
    def cx(self) -> float:
        return float(self.kf.x[0])

    @property
    def cy(self) -> float:
        return float(self.kf.x[1])

    def box(self) -> tuple[float, float, float, float]:
        return (self.cx - self.w / 2, self.cy - self.h / 2,
                self.cx + self.w / 2, self.cy + self.h / 2)

    def _patch(self, gray, cx=None, cy=None):
        return sample_patch(gray, self.cx if cx is None else cx, self.cy if cy is None else cy,
                            self.w * _PADDING, self.h * _PADDING)

    def position_at(self, t: float) -> tuple[float, float]:
        """Where this track was at time ``t`` (nearest history sample)."""
        best = min(self.history, key=lambda s: abs(s[0] - t))
        return best[1], best[2]

    def reanchor(self, gray, box, payload: dict, t_det: float, t_now: float) -> float:
        """Correct the track from a detection of an older frame. Returns how far
        off the track was at that frame, in box sizes (the tracking error)."""
        x1, y1, x2, y2 = box
        dcx, dcy = (x1 + x2) / 2, (y1 + y2) / 2
        hx, hy = self.position_at(t_det)
        err = float(np.hypot(dcx - hx, dcy - hy) / max(self.w, self.h))
        self.kf.shift(dcx - hx, dcy - hy)
        self.w, self.h = max(4.0, x2 - x1), max(4.0, y2 - y1)
        self.payload = payload
        self.state, self.lost_since, self.misses = TRACKING, None, 0
        self.filter = Mosse(self._patch(gray))
        self.history.append((t_now, self.cx, self.cy))
        return err


def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class LiveBoxTracker:
    """Follows detector boxes frame to frame. Single-threaded: call step() and
    correct() from the same thread. Times are seconds of video (frame pts)."""

    PSR_TRACK = 8.0        # >= : confident — adapt appearance
    PSR_LOST = 4.5         # <  : no believable match
    _MAX_STEP = 0.6        # largest believable move per frame, in box sizes
    _COAST = 0.35          # seconds a LOST track keeps being drawn on prediction
    _MAX_MISSES = 2        # detection rounds without it before a track ends
    _MATCH_IOU = 0.2
    _FREEZE_HIDDEN = True  # zero a hidden track's velocity (see step)
    _MATCH_SAME_ID = 2.5   # an identity match is believed up to this many box sizes away

    def __init__(self, identity_key: Optional[str] = None):
        """``identity_key``: payload key naming who a detection is (faces carry
        one from the identity bank). A detection matches the track with the same
        identity first, by overlap only otherwise — so a track that drifted off
        its subject is corrected rather than duplicated."""
        self._identity_key = identity_key
        self._tracks: list[_Track] = []
        self._next_id = 0
        self._gray: Optional[np.ndarray] = None
        self._t: Optional[float] = None
        # Tracking error measured at each correction, in box sizes — what the
        # box would have been off by without it is in held_errors.
        self.errors: deque = deque(maxlen=500)
        self.held_errors: deque = deque(maxlen=500)

    def reset(self) -> None:
        self._tracks.clear()
        self._t = None

    def step(self, gray: np.ndarray, t: float) -> None:
        """Advance every track to this frame."""
        if self._t is not None and (t < self._t or t - self._t > 1.0):
            self.reset()                 # a seek: positions from before it mean nothing
        dt = 1 / 30 if self._t is None else max(1e-3, t - self._t)
        self._gray, self._t = gray, t
        for tr in self._tracks:
            tr.kf.predict(dt)
            px, py = tr.cx, tr.cy
            sx = tr.w * _PADDING / _PATCH
            sy = tr.h * _PADDING / _PATCH
            dx, dy, psr = tr.filter.respond(tr._patch(gray, px, py))
            mx, my = px + dx * sx, py + dy * sy
            tr.psr = psr
            believable = np.hypot(mx - px, my - py) <= self._MAX_STEP * max(tr.w, tr.h)
            # Hysteresis: a lost track needs a confident match to come back.
            # Over plain background the PSR wanders around PSR_LOST by chance,
            # and without this a gone subject flickers back into view.
            floor = self.PSR_TRACK if tr.state == LOST else self.PSR_LOST
            if psr >= floor and believable:
                tr.kf.update(mx, my)
                tr.lost_since = None
                if psr >= self.PSR_TRACK:
                    tr.state = TRACKING
                    tr.filter.adapt(tr._patch(gray))
                else:
                    tr.state = UNCERTAIN
            else:
                if tr.state != LOST:
                    tr.lost_since = t
                tr.state = LOST
                if self._FREEZE_HIDDEN and t - tr.lost_since > self._COAST:
                    tr.kf.x[2:] = 0.0    # hidden now: stop drifting on the last velocity
            tr.history.append((t, tr.cx, tr.cy))

    def correct(self, detections: list[tuple[tuple, dict]], t_det: float) -> None:
        """Fold in a detector pass over the frame at ``t_det``: [(box, payload)].
        Boxes are in the same pixel space as the frames given to step()."""
        if self._gray is None or self._t is None:
            return
        if t_det > self._t + 0.5 or t_det < self._t - 3.0:
            return                       # from before a seek, or wildly out of step
        key = self._identity_key
        pairs = []
        for ti, tr in enumerate(self._tracks):
            hx, hy = tr.position_at(t_det)
            then = (hx - tr.w / 2, hy - tr.h / 2, hx + tr.w / 2, hy + tr.h / 2)
            tid = tr.payload.get(key) if key else None
            for di, (box, p) in enumerate(detections):
                iou = _iou(then, box)
                if tid is not None and p.get(key) == tid:
                    dist = np.hypot((box[0] + box[2]) / 2 - hx, (box[1] + box[3]) / 2 - hy)
                    # A lost track's position is a guess; its identity isn't.
                    if tr.state == LOST or dist <= self._MATCH_SAME_ID * max(tr.w, tr.h):
                        pairs.append((1.0 + iou, ti, di))   # identity outranks any overlap
                        continue
                if iou >= self._MATCH_IOU:
                    pairs.append((iou, ti, di))
        pairs.sort(reverse=True)
        used_t, used_d = set(), set()
        for _iou_v, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            tr = self._tracks[ti]
            held = tr.payload.get("_held_box")
            box, payload = detections[di]
            if held is not None:
                hc = ((held[0] + held[2]) / 2, (held[1] + held[3]) / 2)
                dc = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                self.held_errors.append(float(np.hypot(dc[0] - hc[0], dc[1] - hc[1])
                                              / max(box[2] - box[0], box[3] - box[1], 1)))
            self.errors.append(tr.reanchor(self._gray, box, dict(payload, _held_box=box),
                                           t_det, self._t))
        keep = []
        for ti, tr in enumerate(self._tracks):
            if ti not in used_t:
                tr.misses += 1
                if tr.misses > self._MAX_MISSES:
                    continue
            keep.append(tr)
        self._tracks = keep
        for di, (box, payload) in enumerate(detections):
            if di not in used_d:
                self._tracks.append(_Track(self._next_id, self._gray, box,
                                           dict(payload, _held_box=box), self._t))
                self._next_id += 1

    def boxes(self) -> list[dict]:
        """Boxes to draw now: [{"bbox", "state", "psr", "track_id", **payload}]."""
        out = []
        for tr in self._tracks:
            if tr.state == LOST and tr.lost_since is not None \
                    and self._t - tr.lost_since > self._COAST:
                continue
            d = {k: v for k, v in tr.payload.items() if not k.startswith("_")}
            d.update(bbox=tr.box(), state=tr.state, psr=tr.psr, track_id=tr.id)
            out.append(d)
        return out
