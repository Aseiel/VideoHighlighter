"""How long will training take? Said before the button is pressed.

A person deciding whether to start a run needs one sentence — "about 4
minutes on your graphics card" — not a progress bar that only becomes honest
after the first epoch. This module produces that sentence's number.

Where the number comes from, in order of trust:

1. **This computer's own history.** Every finished run records how many
   images per second it actually trained and validated at, per device kind and
   model size (``ThroughputStore``). The second estimate a person sees is
   therefore measured, not guessed, and says so.
2. **A table of typical speeds** (``TYPICAL``) for a first run. Some rows were
   measured, and are marked; the rest are rough, and the estimate carries a
   wider range for them.

The estimate covers everything the person waits for, not just the loop:
pulling frames out of their videos, the one-time pretrained download, model
setup, every epoch of training plus validation, and the export at the end.

No torch, no Qt: the arithmetic a promise rests on is tested on its own.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from typing import Optional

# Images per second, (train, validate), at 416x416, batch 8, loading included
# where measured end to end. Keyed by (device kind, model size).
#
# Measured 2026-09-17 with synthetic batches (loading excluded, so the loop
# figures are corrected down ~15% below for the dataloader's share):
#   Intel Arc A750, torch 2.8.0+xpu: nano 34.0/42.1, tiny 31.4/40.6, s 29.3/39.2
#   Ryzen Zen 3, 12 threads, CPU:    nano 14.5/42.8, tiny  8.5/24.5, s  5.7/17.1
# CUDA and DirectML rows are not measured here; they are deliberately
# conservative placeholders until a run on such a machine records its own.
TYPICAL = {
    ("xpu", "nano"): (29.0, 36.0),
    ("xpu", "tiny"): (27.0, 34.0),
    ("xpu", "s"): (25.0, 33.0),
    ("cpu", "nano"): (12.0, 36.0),
    ("cpu", "tiny"): (7.0, 21.0),
    ("cpu", "s"): (4.8, 14.5),
    ("cuda", "nano"): (45.0, 90.0),
    ("cuda", "tiny"): (40.0, 80.0),
    ("cuda", "s"): (32.0, 65.0),
    ("dml", "nano"): (14.0, 30.0),
    ("dml", "tiny"): (12.0, 26.0),
    ("dml", "s"): (9.0, 20.0),
}
MEASURED_KINDS = {"xpu", "cpu"}     # which TYPICAL rows came from a real device

# Fixed costs, in seconds.
SECONDS_PER_FRAME_EXTRACT = 0.12    # seek + decode + write one JPEG
SETUP_SECONDS = 15.0                # import torch/yolox, build model, load weights
EXPORT_SECONDS = 20.0               # ONNX export + IR conversion
PRETRAINED_DOWNLOAD_SECONDS = 20.0  # ~20-70 MB once, on a typical connection

# A model trained on less than this rarely finds anything; said before the run.
MIN_USEFUL_FRAMES = 30


def device_kind(device: str) -> str:
    """Collapse a torch device string into the kinds speeds are kept per."""
    d = str(device or "cpu").lower()
    if d.startswith("xpu"):
        return "xpu"
    if d.startswith("cuda"):
        return "cuda"
    if d.startswith("privateuseone") or d.startswith("dml"):
        return "dml"
    return "cpu"


def friendly_device(device: str, name: str = "") -> str:
    """"your Intel Arc A750" / "the processor" — how a person names it."""
    kind = device_kind(device)
    if kind == "cpu":
        return "the processor"
    if name:
        return f"your {name}"
    return {"xpu": "your Intel graphics card", "cuda": "your NVIDIA graphics card",
            "dml": "your graphics card (DirectML)"}.get(kind, "your graphics card")


def friendly_duration(seconds: float) -> str:
    """"about 4 minutes" — rounded the way a person would say it."""
    seconds = max(0.0, float(seconds or 0))
    if seconds < 60:
        return "under a minute"
    minutes = seconds / 60
    if minutes < 10:
        return f"about {max(1, round(minutes))} minute{'s' if round(minutes) != 1 else ''}"
    if minutes < 60:
        return f"about {int(5 * round(minutes / 5))} minutes"
    hours = minutes / 60
    return f"about {hours:.1f} hours".replace(".0 hours", " hours")


# --------------------------------------------------------------------------- #
# What this computer has measured
# --------------------------------------------------------------------------- #
class ThroughputStore:
    """Measured training speeds on this computer, as user data.

    Blended rather than overwritten, so one run on a busy machine does not
    swing the next promise by half.
    """

    BLEND = 0.5

    def __init__(self, path: str):
        self.path = path
        self.rows: dict = {}

    @staticmethod
    def key(kind: str, size: str, image_size) -> str:
        h, w = (tuple(image_size) + (416, 416))[:2] if image_size else (416, 416)
        return f"{kind}/{size}/{int(h)}x{int(w)}"

    def load(self) -> "ThroughputStore":
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                self.rows = {k: v for k, v in data.items() if isinstance(v, dict)}
        except (OSError, ValueError):
            self.rows = {}
        return self

    OVERHEADS = "_overheads"

    def overheads(self) -> dict:
        """Measured fixed costs: ``extract_per_frame`` and ``fixed`` (setup +
        export) seconds. Empty until a run has recorded them."""
        row = self.rows.get(self.OVERHEADS) or {}
        return {k: float(v) for k, v in row.items()
                if k in ("extract_per_frame", "fixed") and isinstance(v, (int, float))}

    def record_overheads(self, extract_per_frame: Optional[float] = None,
                         fixed: Optional[float] = None) -> None:
        row = dict(self.rows.get(self.OVERHEADS) or {})
        for key, value in (("extract_per_frame", extract_per_frame), ("fixed", fixed)):
            if value is None or not math.isfinite(value) or value <= 0:
                continue
            old = row.get(key)
            row[key] = round(value if old is None else
                             self.BLEND * value + (1 - self.BLEND) * float(old), 4)
        if row:
            self.rows[self.OVERHEADS] = row

    def get(self, kind: str, size: str, image_size=(416, 416)) -> Optional[tuple]:
        row = self.rows.get(self.key(kind, size, image_size))
        if not row:
            return None
        try:
            return float(row["train_ips"]), float(row["val_ips"])
        except (KeyError, TypeError, ValueError):
            return None

    def record(self, kind: str, size: str, image_size, train_ips: float,
               val_ips: Optional[float] = None) -> None:
        if not (train_ips and train_ips > 0 and math.isfinite(train_ips)):
            return
        k = self.key(kind, size, image_size)
        old = self.rows.get(k)
        if old:
            train_ips = self.BLEND * train_ips + (1 - self.BLEND) * float(old["train_ips"])
            if val_ips and val_ips > 0:
                val_ips = self.BLEND * val_ips + (1 - self.BLEND) * float(old["val_ips"])
            else:
                val_ips = float(old["val_ips"])
        if not (val_ips and val_ips > 0):
            val_ips = train_ips * 2.5
        self.rows[k] = {"train_ips": round(train_ips, 2), "val_ips": round(val_ips, 2),
                        "runs": int((old or {}).get("runs", 0)) + 1,
                        "updated": time.strftime("%Y-%m-%d")}

    def save(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.rows, fh, indent=2)
        os.replace(tmp, self.path)


# --------------------------------------------------------------------------- #
# The estimate
# --------------------------------------------------------------------------- #
@dataclass
class Estimate:
    seconds: float
    low: float
    high: float
    measured: bool          # from this computer's own runs
    train_frames: int
    val_frames: int
    breakdown: dict

    def sentence(self, device_phrase: str) -> str:
        """The line shown next to the Train button."""
        span = friendly_duration(self.seconds)
        basis = ("measured on this computer" if self.measured
                 else "a first guess; it gets exact after one run")
        text = f"Training will take {span} on {device_phrase} ({basis})."
        if self.train_frames + self.val_frames < MIN_USEFUL_FRAMES:
            text += (f" Only {self.train_frames + self.val_frames} frames so far — "
                     f"a model usually needs {MIN_USEFUL_FRAMES}+ to find anything.")
        return text


def estimate(train_frames: int, val_frames: int, epochs: int, size: str,
             device: str, image_size=(416, 416),
             store: Optional[ThroughputStore] = None,
             pretrained_cached: bool = True) -> Estimate:
    """Seconds for the whole run, with a range. See the module docstring."""
    kind = device_kind(device)
    measured = store.get(kind, size, image_size) if store is not None else None
    if measured:
        train_ips, val_ips = measured
        spread = 0.15
    else:
        train_ips, val_ips = TYPICAL.get((kind, size)) or TYPICAL[("cpu", "tiny")]
        spread = 0.25 if kind in MEASURED_KINDS else 0.6
        # The table is for 416; cost grows with pixels.
        h, w = tuple(image_size)[:2]
        scale = (h * w) / (416 * 416)
        train_ips, val_ips = train_ips / scale, val_ips / scale

    epochs = max(1, int(epochs))
    train_frames, val_frames = max(0, int(train_frames)), max(0, int(val_frames))
    per_epoch = train_frames / max(train_ips, 1e-6) + val_frames / max(val_ips, 1e-6)
    over = store.overheads() if store is not None else {}
    extract = over.get("extract_per_frame", SECONDS_PER_FRAME_EXTRACT)
    fixed = over.get("fixed", SETUP_SECONDS + EXPORT_SECONDS)
    breakdown = {
        "frames": (train_frames + val_frames) * extract,
        "setup_and_export": fixed + (0.0 if pretrained_cached else PRETRAINED_DOWNLOAD_SECONDS),
        "training": per_epoch * epochs,
    }
    total = sum(breakdown.values())
    return Estimate(seconds=total, low=total * (1 - spread), high=total * (1 + spread),
                    measured=bool(measured), train_frames=train_frames,
                    val_frames=val_frames, breakdown=breakdown)


def frames_in_store(store, val_fraction: Optional[float] = None) -> tuple:
    """(train, val) frame counts a ``modules.vision.label_store.LabelStore`` will
    produce, split the way ``build_dataset`` splits it."""
    from modules.vision.label_store import segments, split_segments, DEFAULT_VAL_FRACTION
    usable = store.accepted() + store.negatives()
    if not usable:
        return 0, 0
    frac = DEFAULT_VAL_FRACTION if val_fraction is None else val_fraction
    train_groups, val_groups = split_segments(segments(usable), val_fraction=frac, seed=0)
    count = lambda groups: len({(b.video, b.time) for g in groups for b in g})  # noqa: E731
    return count(train_groups), count(val_groups)


def default_store_path() -> str:
    try:
        from modules.system.app_paths import user_data_dir
        base = user_data_dir()
    except Exception:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "cache", "training_speed.json")
