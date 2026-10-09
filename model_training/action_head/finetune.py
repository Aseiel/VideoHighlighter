"""Fine-tuning the image tower's top blocks together with the action head (LP-FT).

The frozen trainer leaves the encoder as it is and trains only the head. Here
the top ``blocks`` transformer blocks of SigLIP2's image tower are trained too,
starting from a frozen head already trained on the same clips ("linear probe,
then fine-tune"), so the tower's first steps follow a head that is already
right rather than a random one. Measured on held-out source videos
(``docs/plans/2026-10-08-overfitting-and-siglip2-fine-tune.md``): top-1 0.533
frozen -> 0.603 with the top 4 blocks at lr 1e-4.

The recipe is that measurement's, unchanged:

- **Layer-wise learning rates.** Block ``d`` below the top trains at
  ``lr * decay ** d``; the tower's final norm and pooling head at ``lr``; the
  action head at ``head_lr``. AdamW (weight decay 0.05), 6 % warm-up then
  cosine, gradient norm clipped at 1, bf16 autocast where the card has it.
- **The head's own loss**: the same class weights, label smoothing and
  per-clip weights as ``head.train_head``.
- **Views.** Each training clip is seen as 4 of its 12 cached frames, picked
  at random and kept in time order, with one crop (70-100 % of the area,
  aspect within 15 %), flip and brightness/contrast change for all four.
  Scoring always uses the 4 frames the app uses.
- The blocks below the trained ones run without gradients, and above 6
  trained blocks the trained ones are checkpointed, so 8 GB holds a batch of
  16 clips.

Graphics cards only: Intel (XPU) or NVIDIA (CUDA). DirectML has no bf16
autocast and the transformer backward pass on it is untested, so it is
refused with a sentence; the processor is used only when asked for.
"""
from __future__ import annotations

import contextlib
import copy
import math
import sys
import time
from dataclasses import asdict, dataclass
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F

from model_training.action_head import frames as FR
from model_training.action_head import head as H

WEIGHT_DECAY = 0.05
WARMUP = 0.06
CLIP_NORM = 1.0
CHECKPOINT_ABOVE = 6        # trained blocks above which they are checkpointed
CROP_SCALE = (0.7, 1.0)
CROP_LOG_RATIO = 0.15
BRIGHTNESS = 0.15
CONTRAST = 0.15
PREDICT_BATCH = 32
START_MIN_COSINE = 0.99     # torch tower vs the app's encoder, before training


class DeviceError(RuntimeError):
    """This machine cannot fine-tune; the message says why, for the log."""


class OutOfMemory(RuntimeError):
    """The card ran out of memory; the message says what to change."""


@dataclass
class Settings:
    blocks: int = 4
    epochs: int = 10
    lr: float = 1e-4
    decay: float = 0.8
    head_lr: float = 2e-4
    batch: int = 16
    seed: int = 0


# ── device ───────────────────────────────────────────────────────────────────

def resolve_device(requested: str = "auto") -> str:
    """``auto`` -> XPU, else CUDA; ``cpu`` only when asked. Raises
    DeviceError with a sentence otherwise."""
    req = (requested or "auto").strip().lower()
    if req in ("dml", "directml") or req.startswith("privateuseone"):
        raise DeviceError(
            "Fine-tuning does not run on DirectML (no bf16, and the image model's training "
            "is untested there). Train without --finetune-blocks: the frozen model trains on "
            "any computer.")
    if req == "cpu":
        return "cpu"
    xpu = hasattr(torch, "xpu") and torch.xpu.is_available()
    cuda = torch.cuda.is_available()
    if req == "auto":
        if xpu:
            return "xpu"
        if cuda:
            return "cuda"
        raise DeviceError(
            "Fine-tuning the image model needs an Intel Arc (XPU) or NVIDIA (CUDA) graphics "
            "card that PyTorch can use, and none was found. Train without --finetune-blocks, "
            "or pass --device cpu to run it on the processor (very slow).")
    if req.startswith("xpu") and xpu or req.startswith("cuda") and cuda:
        return req
    raise DeviceError(f"PyTorch cannot use the device {requested!r} here")


def autocast_dtype(device: str):
    """bf16 where the device does it, else None (fp32)."""
    try:
        if device.startswith("xpu"):
            return torch.bfloat16 if torch.xpu.is_bf16_supported() else None
        if device.startswith("cuda"):
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else None
    except Exception:  # noqa: BLE001
        return None
    return None


def _autocast(device: str):
    dtype = autocast_dtype(device)
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device.split(":")[0], dtype=dtype)


def empty_cache(device: str) -> None:
    try:
        if device.startswith("xpu"):
            torch.xpu.empty_cache()
        elif device.startswith("cuda"):
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def is_out_of_memory(exc: BaseException) -> bool:
    return isinstance(exc, getattr(torch, "OutOfMemoryError", ())) or (
        isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower())


@contextlib.contextmanager
def keep_awake():
    """Ask Windows not to sleep while this runs (a fine-tune takes about an
    hour), and stop asking after. Per thread, so it is set and cleared on the
    thread doing the training."""
    flags = None
    if sys.platform == "win32":
        try:
            import ctypes
            set_state = ctypes.windll.kernel32.SetThreadExecutionState
            # ES_CONTINUOUS | ES_SYSTEM_REQUIRED: no idle sleep, the screen may still go off.
            set_state(0x80000000 | 0x00000001)
            flags = set_state
        except Exception:  # noqa: BLE001 - only costs a sleeping computer
            flags = None
    try:
        yield
    finally:
        if flags is not None:
            try:
                flags(0x80000000)
            except Exception:  # noqa: BLE001
                pass


# ── the tower ────────────────────────────────────────────────────────────────

def load_base_tower(source: str, revision: str):
    """The pinned checkpoint's image tower (a ``SiglipVisionTransformer``),
    fp32 on the processor. The first call downloads the checkpoint into the
    Hugging Face cache; later ones work offline."""
    from transformers import AutoModel
    model = AutoModel.from_pretrained(source, revision=revision)
    tower = model.vision_model
    del model
    return tower.float().eval()


def _layer(layer, h):
    out = layer(h, None)
    return out[0] if isinstance(out, tuple) else out


class Tower(torch.nn.Module):
    """The image tower with only its top ``blocks`` blocks (and its final
    norm and pooling head) trainable; the blocks below run without
    gradients. Frames [N, 3, S, S] -> pooled vectors [N, dims]."""

    def __init__(self, vm, blocks: int):
        super().__init__()
        self.vm = vm
        n = len(vm.encoder.layers)
        self.blocks = max(0, min(int(blocks), n))
        for p in vm.parameters():
            p.requires_grad_(False)
        for p in self.trained_parameters():
            p.requires_grad_(True)
        self.checkpoint = self.blocks > CHECKPOINT_ABOVE

    def top_layers(self):
        layers = self.vm.encoder.layers
        return list(layers[len(layers) - self.blocks:]) if self.blocks else []

    def trained_parameters(self):
        if not self.blocks:
            return []
        out = [p for layer in self.top_layers() for p in layer.parameters()]
        out += list(self.vm.post_layernorm.parameters())
        if getattr(self.vm, "head", None) is not None:
            out += list(self.vm.head.parameters())
        return out

    def forward(self, x):
        layers = self.vm.encoder.layers
        n = len(layers)
        with torch.no_grad():
            h = self.vm.embeddings(x)
            for layer in layers[:n - self.blocks]:
                h = _layer(layer, h)
        for layer in layers[n - self.blocks:]:
            if self.checkpoint and self.training:
                h = torch.utils.checkpoint.checkpoint(_layer, layer, h, use_reentrant=False)
            else:
                h = _layer(layer, h)
        h = self.vm.post_layernorm(h)
        head = getattr(self.vm, "head", None)
        return head(h) if head is not None else h[:, 0]


class Vision(torch.nn.Module):
    """The exported form: ``pixel_values`` -> ``image_embeds`` (the pooled
    vector), as tools/export_frame_encoder.py exports the shared encoder."""

    def __init__(self, vm):
        super().__init__()
        self.vm = vm

    def forward(self, pixel_values):
        return self.vm(pixel_values=pixel_values).pooler_output


@torch.no_grad()
def tower_vector(vm, pixels: np.ndarray, device: str = "cpu") -> np.ndarray:
    """fp32 pooled vectors of a tower for preprocessed pixels [N, 3, S, S]."""
    vm = vm.to(device).eval()
    return Vision(vm)(torch.as_tensor(pixels, device=device)).float().cpu().numpy()


def start_check(vm, shared_probe: np.ndarray, device: str) -> float:
    """Cosine between this tower's vector for ``probe_pixels()`` and the
    shared encoder's: proof the fine-tune starts from the app's encoder.
    Raises ValueError when they differ."""
    from modules.vision import frame_encoder
    got = tower_vector(vm, frame_encoder.probe_pixels(), device)[0]
    cos = frame_encoder._cosine(got, shared_probe)
    if not cos >= START_MIN_COSINE:
        raise ValueError(f"the starting weights are not the app's encoder (cosine {cos:.4f}); "
                         f"is the Hugging Face cache holding another revision?")
    return cos


# ── frames and augmentation ──────────────────────────────────────────────────

def to_device_pixels(u8: np.ndarray, device: str) -> torch.Tensor:
    """Cached uint8 frames [..., S, S, 3] -> float [..., 3, S, S] in [-1, 1] on
    the device; equal to ``frames.pixels`` (``x / 127.5 - 1`` is
    ``(x / 255 - 0.5) / 0.5``)."""
    t = torch.from_numpy(np.ascontiguousarray(u8)).to(device, non_blocking=True)
    return t.movedim(-1, -3).float() / 127.5 - 1.0


def pick_slots(rng, n: int, k: int = FR.SCORING) -> np.ndarray:
    """For each of ``n`` clips, ``k`` of the 12 cached slots at random, in
    time order: [n, k]."""
    out = np.empty((n, k), int)
    for i in range(n):
        s = rng.choice(FR.SLOTS, k, replace=False)
        out[i] = s[np.argsort(FR.SLOT_POSITIONS[s], kind="stable")]
    return out


def augment(px: torch.Tensor, rng) -> torch.Tensor:
    """px [B, T, 3, S, S]: one crop, flip and colour change per clip, the same
    for all its frames."""
    B, T, C, S, _ = px.shape
    out = torch.empty_like(px)
    for b in range(B):
        scale = rng.uniform(*CROP_SCALE)
        ratio = math.exp(rng.uniform(-CROP_LOG_RATIO, CROP_LOG_RATIO))
        w = int(round(S * min(1.0, math.sqrt(scale * ratio))))
        h = int(round(S * min(1.0, math.sqrt(scale / ratio))))
        x0, y0 = int(rng.integers(0, S - w + 1)), int(rng.integers(0, S - h + 1))
        v = px[b, :, :, y0:y0 + h, x0:x0 + w]
        v = F.interpolate(v, size=(S, S), mode="bilinear", align_corners=False)
        if rng.random() < 0.5:
            v = v.flip(-1)
        bright = rng.uniform(-BRIGHTNESS, BRIGHTNESS)
        contrast = rng.uniform(1 - CONTRAST, 1 + CONTRAST)
        m = v.mean(dim=(1, 2, 3), keepdim=True)
        out[b] = ((v - m) * contrast + m + bright).clamp(-1, 1)
    return out


# ── training ─────────────────────────────────────────────────────────────────

def param_groups(tower: Tower, head, s: Settings) -> list:
    """Learning-rate groups: each trained block at ``lr * decay ** depth``
    (0 = the top block), the final norm and pooling head at ``lr``, the
    action head at ``head_lr``."""
    groups = []
    top = tower.top_layers()
    for j, layer in enumerate(top):
        depth = len(top) - 1 - j
        groups.append({"params": list(layer.parameters()), "lr": s.lr * s.decay ** depth})
    rest = [p for p in tower.trained_parameters()
            if not any(p is q for g in groups for q in g["params"])]
    if rest:
        groups.append({"params": rest, "lr": s.lr})
    groups.append({"params": list(head.parameters()), "lr": s.head_lr})
    return groups


def lr_factor(step: int, total: int) -> float:
    warm = max(1, int(WARMUP * total))
    if step < warm:
        return step / warm
    return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))


@torch.no_grad()
def predict(tower: Tower, head, cache: FR.FrameCache, rows, device: str,
            batch: int = PREDICT_BATCH) -> np.ndarray:
    """Each action's score for clips ``rows`` of the cache, from the 4 frames
    the app scores: [N, classes]."""
    tower.eval()
    head.eval()
    out = []
    rows = np.asarray(rows)
    for i in range(0, len(rows), batch):
        r = rows[i:i + batch]
        px = to_device_pixels(cache.read(r, slice(0, FR.SCORING)), device)
        with _autocast(device):
            v = tower(px.flatten(0, 1))
        v = v.float().view(len(r), FR.SCORING, -1)
        out.append(torch.sigmoid(head(v)).float().cpu())
    if not out:
        return np.zeros((0, head.centres.shape[0]), np.float32)
    return torch.cat(out).numpy()


@torch.no_grad()
def vectors(tower: Tower, cache: FR.FrameCache, rows, device: str,
            batch: int = PREDICT_BATCH) -> np.ndarray:
    """The tower's fp32 vectors for the 4 scoring frames of ``rows``:
    [N, 4, dims]."""
    tower.eval()
    out = []
    rows = np.asarray(rows)
    for i in range(0, len(rows), batch):
        r = rows[i:i + batch]
        px = to_device_pixels(cache.read(r, slice(0, FR.SCORING)), device)
        out.append(tower(px.flatten(0, 1)).float().view(len(r), FR.SCORING, -1).cpu())
    return torch.cat(out).numpy()


def finetune(base_vm, head0, cache: FR.FrameCache, rows, targets: np.ndarray,
             s: Settings, device: str, *, log: Callable[[str], None] = print,
             check: Callable[[], None] = lambda: None,
             eval_fn: Optional[Callable] = None, label: str = ""):
    """LP-FT from the frozen head ``head0`` on clips ``rows`` of the cache.

    ``base_vm`` is the starting tower (left untouched: a copy is trained).
    ``eval_fn(tower, head)`` runs after each epoch and returns
    ``(anything, accuracy)``; its results are returned in order. ``check`` is
    called every batch and may raise to stop. Returns
    ``(tower, head, results)``. Raises OutOfMemory with a sentence.
    """
    torch.manual_seed(s.seed)
    rng = np.random.default_rng(s.seed)
    rows = np.asarray(rows)
    tower = Tower(copy.deepcopy(base_vm), s.blocks).to(device)
    head = copy.deepcopy(head0).to(device)
    groups = param_groups(tower, head, s)
    opt = torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)
    for g in opt.param_groups:
        g["base_lr"] = g["lr"]
    trained = [p for g in groups for p in g["params"]]

    n_classes = targets.shape[1]
    yt = torch.as_tensor(np.asarray(targets, np.float32))
    w = torch.as_tensor(H.class_weights(np.asarray(targets, np.float32)), dtype=torch.float32)
    sample_w = ((yt * w).sum(1) / yt.sum(1).clamp(min=1)).to(device)
    smooth = (yt * (1 - H.LABEL_SMOOTHING) + H.LABEL_SMOOTHING / n_classes).to(device)
    batch = max(1, min(s.batch, len(rows)))
    per_epoch = max(1, len(rows) // batch)
    total = per_epoch * s.epochs
    step, results = 0, []
    try:
        for ep in range(s.epochs):
            tower.train()
            head.train()
            started, loss_sum = time.time(), 0.0
            perm = rng.permutation(len(rows))
            for bi in range(per_epoch):
                check()
                b = perm[bi * batch:(bi + 1) * batch]
                slots = pick_slots(rng, len(b))
                fr = cache.read(rows[b])
                fr = np.take_along_axis(fr, slots[:, :, None, None, None], 1)
                px = augment(to_device_pixels(fr, device), rng)
                f = lr_factor(step, total)
                for g in opt.param_groups:
                    g["lr"] = g["base_lr"] * f
                with _autocast(device):
                    v = tower(px.flatten(0, 1))
                v = v.float().view(len(b), FR.SCORING, -1)
                loss = F.binary_cross_entropy_with_logits(head(v), smooth[b], reduction="none")
                loss = (loss.sum(1) * sample_w[b]).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trained, CLIP_NORM)
                opt.step()
                step += 1
                loss_sum += float(loss.detach())
            line = (f"      {label}epoch {ep + 1}/{s.epochs}: loss {loss_sum / per_epoch:.3f}, "
                    f"{time.time() - started:.0f} s")
            if eval_fn is not None:
                res = eval_fn(tower, head)
                results.append(res)
                line += f", held-out {res[1]:.3f}"
            log(line)
    except Exception as e:  # noqa: BLE001 - only memory is translated; the rest goes up
        if is_out_of_memory(e):
            del tower, head, opt
            empty_cache(device)
            raise OutOfMemory(
                f"The graphics card ran out of memory fine-tuning {s.blocks} blocks at a batch "
                f"of {s.batch} clips. Try fewer blocks (--finetune-blocks) or a smaller batch "
                f"(--ft-batch).") from None
        raise
    return tower, head, results


def settings_dict(s: Settings) -> dict:
    return asdict(s)
