"""Train a small YOLOX detector in-process, with progress and cancellation.

``train_yolox.py`` next to this file prepares a dataset and then *prints the
commands* to run training inside a separately-cloned YOLOX repo. That is a
developer's handoff, and three things make it unusable as a feature:

* the clone has to exist, so nothing can train on a machine that only installed
  the app;
* YOLOX's own ``tools/train.py`` is CUDA-bound throughout — ``.cuda()``,
  ``torch.cuda.amp``, nccl — with no device flag, so it cannot run on an Intel
  GPU at all;
* a printed command reports its progress to a terminal nobody is watching, and
  cannot be cancelled by a person who changed their mind.

So this module borrows YOLOX's *model, loss and data pipeline* — which are
device-agnostic and Apache-2.0 — and supplies its own training loop around
them. Every tensor moves with ``.to(device)``, progress is a callback, and
stopping is checked between steps.

Nothing here imports Qt, and ``torch`` and ``yolox`` are imported inside the
functions that need them, so the module can be imported (and its arithmetic
tested) on a machine that has neither.

The dataset it expects
----------------------

COCO format, in the layout ``yolox.data.COCODataset`` reads::

    dataset/
      annotations/train.json      instances, with a "categories" list
      annotations/val.json
      train/<image files>
      val/<image files>

``training/train_yolox_dataset.py`` already produces this from labeller
exports; the miner will produce it directly.

What it returns, and what it does not
-------------------------------------

A finished run leaves weights on disk and reports the validation loss it
reached. It deliberately does **not** report mAP: a real detection metric needs
the COCO evaluator, and until the held-out set of the training loop exists
there is nothing honest to compare a number against. Validation loss is enough
to answer "did this round beat the last one", which is the only question asked
of it today.
"""
from __future__ import annotations

import json
import os
import platform
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

# YOLOX model sizes worth offering. Depth and width are the multipliers YOLOX's
# own experiment files use; "nano" additionally uses depthwise convolutions.
# Bigger than "s" is not a mini model and is not the point of this feature.
SIZES = {
    "nano": (0.33, 0.25, True),
    "tiny": (0.33, 0.375, False),
    "s":    (0.33, 0.50, False),
}

DEFAULT_SIZE = "tiny"
DEFAULT_IMAGE_SIZE = (416, 416)

# Pretrained COCO checkpoints, Apache-2.0, from the same YOLOX release
# ``tools/get_yolox_model.py`` already pulls its ONNX from.
#
# **Fine-tuning from these is not an optimisation, it is the feature.** A
# detector trained from random initialisation on the order of a hundred images
# learns nothing: measured here, ten epochs on 48 images left objectness at
# 0.5005 mean — sigmoid(0), i.e. the network answering "no idea" everywhere —
# and the exported model produced zero detections at any threshold. Starting
# from COCO weights means the backbone already knows edges, texture and
# objectness, and the run only has to learn a few classes on top. It is also
# what the user's earlier ultralytics run did (`model: yolo11n.pt`), which is
# why that one worked at ~100 samples and this one did not.
PRETRAINED_BASE = (
    "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0"
)

# Resolved against the repository, not the working directory. A relative path
# here caches the download wherever the process happened to be started from,
# so the same 40 MB is fetched again for every caller with a different cwd —
# and the copy the tests look for is somewhere nobody thought to look.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRETRAINED_DIR = os.path.join(_REPO_ROOT, "models", "yolox", "pretrained")


class Cancelled(Exception):
    """Raised inside the loop when ``should_stop()`` first returns True.

    A distinct type rather than a bare return so a caller can tell "the user
    stopped this" from "training finished" without inspecting the result.
    """


@dataclass
class Progress:
    """One report from inside the loop.

    Carries both the fine-grained position (for a bar that moves) and the
    epoch's summary (for a line that means something), because a UI needs the
    first to look alive and the second to be worth reading.
    """

    epoch: int                 # 1-based
    total_epochs: int
    step: int                  # 1-based, within this epoch
    steps_per_epoch: int
    loss: float
    elapsed: float             # seconds since the run started
    eta: float                 # seconds remaining, 0.0 when not yet knowable
    stage: str = "train"       # "train" | "validate" | "save"

    @property
    def fraction(self) -> float:
        """Overall completion in [0,1] — epochs and steps together."""
        total = max(1, self.total_epochs * self.steps_per_epoch)
        done = (self.epoch - 1) * self.steps_per_epoch + self.step
        return min(1.0, done / total)


@dataclass
class TrainingResult:
    """What a finished run produced."""

    weights_path: str
    labels_path: str
    class_names: list
    epochs_completed: int
    best_val_loss: float
    final_train_loss: float
    seconds: float
    device: str
    history: list = field(default_factory=list)   # [{epoch, train_loss, val_loss}]
    # Measured speed, loading included — what the next estimate is built from.
    train_images_per_second: float = 0.0
    val_images_per_second: float = 0.0
    loop_seconds: float = 0.0      # epochs only; ``seconds`` minus this is setup

    def summary(self) -> str:
        # ASCII only, for the same reason the run's prints are: this string
        # reaches consoles whose codepage is not UTF-8.
        return (f"{len(self.class_names)} class(es), {self.epochs_completed} epochs "
                f"on {self.device} in {self.seconds / 60:.1f} min, "
                f"best validation loss {self.best_val_loss:.4f}")


@dataclass
class EpochReport:
    """Handed to ``on_epoch`` after each epoch's validation.

    ``detector`` is the model as it stands right now, wrapped so it answers
    ``detect(frame_bgr)`` exactly like the detector the app will install —
    same letterbox, same decode, same NMS. It is only valid inside the
    callback: the loop moves on as soon as the callback returns.
    """

    epoch: int
    total_epochs: int
    train_loss: float
    val_loss: float
    elapsed: float
    eta: float
    detector: object = None


class _LiveDetector:
    """The model mid-training, answering like ``YoloxOpenVINODetector``.

    Borrows that class's pre-processing and decode rather than YOLOX's own
    ``postprocess``: the preview is meant to show what the *installed* model
    will see, so it must go through the same arithmetic, and YOLOX's
    postprocess needs torchvision and an on-device decode that not every
    backend here supports. The head is switched to raw-grid output for the
    call — the layout the export uses — and restored afterwards, along with
    train mode.
    """

    def __init__(self, model, device, class_names, image_size, score_thr=0.3):
        from modules.vision.detection_backend import YoloxOpenVINODetector, NMS_THR
        self._model = model
        self._device = device
        self._impl = YoloxOpenVINODetector.__new__(YoloxOpenVINODetector)
        self._impl.class_names = list(class_names)
        self._impl.score_thr = float(score_thr)
        self._impl.nms_thr = NMS_THR
        self._impl.input_size = (int(image_size[0]), int(image_size[1]))
        self._impl._infer = self._infer

    def _infer(self, blob):
        import torch
        head = self._model.head
        was_decoding, was_training = head.decode_in_inference, self._model.training
        head.decode_in_inference = False
        self._model.eval()
        try:
            with torch.no_grad():
                out = self._model(torch.from_numpy(blob).to(self._device))
            return out.float().cpu().numpy()
        finally:
            head.decode_in_inference = was_decoding
            self._model.train(was_training)

    def detect(self, frame_bgr):
        return self._impl.detect(frame_bgr)


# ── pure helpers ──────────────────────────────────────────────────────────
# Kept free of torch so the arithmetic a progress bar depends on can be tested
# without a GPU, a dataset, or an install.

def eta_seconds(done: int, total: int, elapsed: float) -> float:
    """Seconds remaining, from a linear extrapolation of what has run.

    Zero until at least one unit is done — an estimate from no evidence is a
    number that swings wildly in the first seconds, which reads as a broken
    progress bar rather than an honest unknown.
    """
    if done <= 0 or total <= 0 or elapsed <= 0 or done >= total:
        return 0.0
    return (elapsed / done) * (total - done)


def directml_device(requested=None):
    """The DirectML device string, or None. Same reasoning as the probes above:
    ``modules.system.directml_device`` imports only ``os`` and ``typing``, so using it
    costs nothing this module was protecting, and the DirectML footguns are
    written down once instead of once per caller."""
    try:
        from modules.system import directml_device as dml
    except Exception:  # noqa: BLE001 — absent module means "no DirectML"
        return None
    try:
        return dml.normalize(requested) if requested is not None else dml.device_string()
    except Exception as e:  # noqa: BLE001 — a probe must never break a run
        print(f"[train] DirectML probe failed ({type(e).__name__}: {e})")
        return None


def is_directml(device) -> bool:
    try:
        from modules.system import directml_device as dml
        return dml.is_directml(device)
    except Exception:  # noqa: BLE001
        return False


def resolve_device(requested: str = "AUTO") -> str:
    """Pick a torch device string: an explicit request, else XPU, else CUDA,
    else DirectML, else CPU.

    Intel is probed first, deliberately. This is the only detector-training
    path in the app, the machine it was written on has an Arc and no NVIDIA
    card, and a helper that reaches for CUDA first answers "cpu" there — which
    would silently turn a twenty-minute run into an overnight one.

    DirectML is probed last, and it is the reason this loop was worth writing
    in the first place. The docstring at the top of this module explains that
    YOLOX's own trainer is unusable here because it is CUDA-bound throughout —
    ``.cuda()``, ``torch.cuda.amp``, nccl. This loop moves every tensor with
    ``.to(device)`` and uses no autocast, which is exactly what DirectML needs
    too, so an AMD card gets training for free from work already done for Arc.

    A local probe rather than ``modules.system.device_utils``, matching the reasoning
    in ``llm/owl_detect.py``: this module is imported lazily, sometimes inside
    a frozen exe, and stays self-contained so a device query cannot drag in
    unrelated machinery.
    """
    requested = (requested or "AUTO").strip()
    if requested and requested.upper() != "AUTO":
        if is_directml(requested):
            return directml_device(requested) or "cpu"
        return requested.lower()
    try:
        import torch
    except Exception:
        return "cpu"
    try:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return "xpu"
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return directml_device() or "cpu"


def default_workers() -> int:
    """Dataloader workers. Zero on Windows.

    Windows has no ``fork``, so every worker re-imports the parent process —
    which here means re-importing torch and yolox per worker, and, in a frozen
    build, re-running the app's entry point. The startup cost exceeds the
    loading it saves on datasets this size.
    """
    return 0 if platform.system() == "Windows" else 4


def read_class_names(annotations_path: str) -> list:
    """Class names from a COCO annotations file, in category-id order.

    The order is the contract with the exported model: the detector reports an
    index, and ``labels.json`` is what turns that back into a name. Sorting by
    id rather than by name is what keeps the two agreeing.
    """
    with open(annotations_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    categories = sorted(data.get("categories", []), key=lambda c: c["id"])
    return [str(c["name"]) for c in categories]


def write_labels_sidecar(path: str, class_names: Sequence) -> str:
    """Write the ``labels.json`` the app reads beside a custom model.

    Load-bearing, not housekeeping: a YOLOX export embeds no class-name
    metadata at all, so without this file the app finds zero names, logs "No
    class names for custom model", and silently falls back to the 80-class
    detector — a failure that looks like a bad model rather than a missing file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump([str(n) for n in class_names], fh, indent=2)
    return path


def pretrained_path(size: str, cache_dir: str = PRETRAINED_DIR) -> str:
    """Where a size's COCO checkpoint is cached."""
    return os.path.join(cache_dir, f"yolox_{size}.pth")


def fetch_pretrained(size: str, cache_dir: str = PRETRAINED_DIR,
                     progress: Optional[Callable] = None) -> str:
    """Download the COCO checkpoint for ``size`` if it is not cached. → path

    Downloaded to a ``.part`` and renamed, so an interrupted fetch cannot leave
    a truncated file that later loads as a corrupt checkpoint — the failure
    would surface as an unintelligible torch error hours later rather than as
    a failed download.
    """
    import urllib.request

    if size not in SIZES:
        raise ValueError(f"no pretrained checkpoint for size {size!r}")
    dest = pretrained_path(size, cache_dir)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest

    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    url = f"{PRETRAINED_BASE}/yolox_{size}.pth"
    print(f"[train] fetching pretrained weights: {url}")

    def _hook(blocks, block_size, total):
        if total > 0 and progress is not None:
            progress(min(blocks * block_size, total), total)

    tmp = dest + ".part"
    urllib.request.urlretrieve(url, tmp, reporthook=_hook)
    os.replace(tmp, dest)
    return dest


def load_pretrained_backbone(model, checkpoint_path: str) -> tuple:
    """Copy COCO weights into ``model``, skipping anything shaped differently.

    The head's class predictors are sized by the number of classes, so they
    never match a user's own class list and must not be forced. Everything else
    — the backbone and the neck, which is where the useful knowledge is —
    transfers unchanged.

    Returns ``(loaded, skipped)`` counts rather than logging alone, so a caller
    can refuse to continue if the overlap is implausibly small. A silent
    near-total mismatch is the failure worth catching: it loads, it trains, and
    it is no better than starting from scratch.
    """
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = checkpoint.get("model", checkpoint)
    target = model.state_dict()

    usable = {k: v for k, v in source.items()
              if k in target and target[k].shape == v.shape}
    skipped = len(source) - len(usable)
    model.load_state_dict(usable, strict=False)
    return len(usable), skipped


# ── the run ───────────────────────────────────────────────────────────────

def _build_exp(dataset_dir: str, num_classes: int, size: str,
               image_size: tuple, batch_size: int, workers: int,
               epochs: int):
    """A YOLOX experiment object pointed at our dataset.

    Subclassing ``Exp`` rather than writing a file on disk: YOLOX's own
    workflow generates an experiment module and passes its path to a CLI, which
    is exactly the indirection this module exists to remove.
    """
    from yolox.exp import Exp

    depth, width, depthwise = SIZES[size]

    class _Exp(Exp):
        def __init__(self):
            super().__init__()
            self.depth = depth
            self.width = width
            self.depthwise = depthwise
            self.num_classes = num_classes
            self.input_size = tuple(image_size)
            self.test_size = tuple(image_size)
            # Multi-scale jitter off: it helps on large datasets and mostly
            # adds variance on the small ones this feature produces.
            self.multiscale_range = 0
            self.data_dir = dataset_dir
            self.train_ann = "train.json"
            self.val_ann = "val.json"
            self.data_num_workers = workers
            # The scheduler reads max_epoch to shape its cosine decay, so this
            # has to be the real epoch count even though our loop, not YOLOX,
            # counts the epochs.
            self.max_epoch = max(1, epochs)
            self.eval_interval = 10 ** 9
            # YOLOX warms up over 5 epochs by default, which is most of a short
            # run: on 20 epochs the model would spend a quarter of its life
            # below its intended learning rate. Scale it down for short runs
            # and leave the default where it was designed to apply.
            self.warmup_epochs = 1 if epochs <= 20 else 5

    exp = _Exp()
    exp.batch_size = batch_size
    return exp


def _loader(exp, dataset_dir: str, split: str, batch_size: int,
            image_size: tuple, workers: int, train: bool):
    """A plain torch DataLoader over a COCO split.

    Built here rather than via ``exp.get_data_loader`` because that helper
    wires in an infinite sampler and distributed plumbing this loop does not
    use, and because an epoch has to be a finite, countable thing for a
    progress bar to mean anything.
    """
    import torch
    from yolox.data import COCODataset, TrainTransform

    # Validation uses TrainTransform with the augmentation switched off, NOT
    # ValTransform. YOLOX's ValTransform returns `np.zeros((1, 5))` as the
    # target for every image — it is built for inference, where the labels are
    # the evaluator's business and the transform's job is only the image. Feed
    # its output to the loss and every frame is scored against "nothing is
    # here": the number looks plausible, moves barely at all between epochs,
    # and measures nothing. Found by a validation loss identical to four
    # decimals across two epochs whose training loss had moved.
    preproc = TrainTransform(
        max_labels=50,
        flip_prob=0.5 if train else 0.0,
        hsv_prob=1.0 if train else 0.0,
    )
    dataset = COCODataset(
        data_dir=dataset_dir,
        json_file=f"{split}.json",
        name=split,
        img_size=tuple(image_size),
        preproc=preproc,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=workers,
        drop_last=False,
        collate_fn=_collate,
    )
    return dataset, loader


def _collate(batch):
    """Stack images and targets, discarding COCODataset's bookkeeping fields.

    ``COCODataset`` yields ``(img, target, img_info, img_id)``; the loss only
    wants the first two, and ``img_info`` is a tuple of varying content that
    the default collate would try, and fail, to stack.
    """
    import numpy as np
    import torch

    images = torch.from_numpy(np.stack([np.ascontiguousarray(b[0]) for b in batch])).float()
    targets = torch.from_numpy(np.stack([b[1] for b in batch])).float()
    return images, targets


def train(dataset_dir: str,
          output_dir: str,
          class_names: Optional[Sequence] = None,
          epochs: int = 30,
          batch_size: int = 8,
          size: str = DEFAULT_SIZE,
          image_size: tuple = DEFAULT_IMAGE_SIZE,
          device: str = "AUTO",
          learning_rate: float = 0.01,
          workers: Optional[int] = None,
          pretrained: bool = True,
          progress: Optional[Callable] = None,
          should_stop: Optional[Callable] = None,
          on_epoch: Optional[Callable] = None) -> TrainingResult:
    """Train a small detector and leave it in ``output_dir``.

    ``progress`` is called with a :class:`Progress` per step; ``should_stop``
    is polled between steps and raises :class:`Cancelled` when it first returns
    True; ``on_epoch`` receives an :class:`EpochReport` after each epoch, with
    a detector for showing what the model finds so far. All plain callables,
    so nothing here depends on a UI framework.

    Raises ``Cancelled`` if stopped, and lets any other failure propagate — a
    training run that swallows its own errors and returns a half-trained model
    is worse than one that stops and says why.
    """
    # Argument checks before the torch import: loading it costs seconds, and a
    # caller who passed a bad size should hear about it immediately.
    if size not in SIZES:
        raise ValueError(f"size must be one of {sorted(SIZES)}, not {size!r}")

    import torch

    started = time.perf_counter()
    workers = default_workers() if workers is None else workers
    device = resolve_device(device)
    if is_directml(device):
        # Registering the backend before the first `.to(device)`. The import is
        # the registration — without it torch does not know what
        # "privateuseone" means and the run dies at the model move, hours of
        # dataset preparation later on a big job.
        try:
            import torch_directml  # noqa: F401 — imported for the side effect
            print(f"[train] DirectML backend registered — training on {device}. "
                  f"Experimental: see docs/AMD-GPU.md")
        except Exception as e:  # noqa: BLE001
            print(f"[train] torch-directml will not import ({type(e).__name__}: "
                  f"{e}); training on the CPU instead")
            device = "cpu"

    train_ann = os.path.join(dataset_dir, "annotations", "train.json")
    if class_names is None:
        class_names = read_class_names(train_ann)
    class_names = [str(n) for n in class_names]
    if not class_names:
        raise ValueError(f"No classes found in {train_ann}")

    def _check_stop():
        if should_stop is not None and should_stop():
            raise Cancelled("training stopped by request")

    exp = _build_exp(dataset_dir, len(class_names), size,
                     image_size, batch_size, workers, epochs)

    _, train_loader = _loader(exp, dataset_dir, "train", batch_size,
                              image_size, workers, train=True)
    val_loader = None
    if os.path.exists(os.path.join(dataset_dir, "annotations", "val.json")):
        _, val_loader = _loader(exp, dataset_dir, "val", batch_size,
                                image_size, workers, train=False)

    model = exp.get_model()
    if pretrained:
        weights = fetch_pretrained(size)
        loaded, skipped = load_pretrained_backbone(model, weights)
        print(f"[train] fine-tuning from COCO weights: {loaded} tensors loaded, "
              f"{skipped} skipped (the class heads, which are sized by your "
              f"class count)")
        if loaded < 50:
            raise RuntimeError(
                f"only {loaded} tensors matched the pretrained checkpoint — "
                f"training would start from near-random weights and produce "
                f"nothing usable at this dataset size. Check that the "
                f"checkpoint matches size={size!r}.")
    else:
        print("[train] training from scratch — expect this to need thousands of "
              "images, not hundreds")
    model = model.to(device)
    model.train()

    # YOLOX's own parameter grouping: no weight decay on norms and biases.
    # Worth taking from the library rather than reimplementing, since getting
    # it wrong degrades quietly rather than failing.
    exp.basic_lr_per_img = learning_rate / max(1, batch_size)
    optimizer = exp.get_optimizer(batch_size)

    steps_per_epoch = max(1, len(train_loader))

    # The learning rate MUST be driven per iteration. `get_optimizer` builds
    # the optimizer at `warmup_lr`, which is 0 by default whenever warmup is
    # enabled, and YOLOX's own trainer raises it from the scheduler on every
    # step. Skip that and the optimizer steps at lr=0 for the entire run: the
    # weights never move, training loss still wanders because augmentation is
    # random, and the only visible symptom is a validation loss that is
    # identical to four decimal places. It cost an afternoon once; the
    # regression test pins it.
    lr_scheduler = exp.get_lr_scheduler(
        exp.basic_lr_per_img * batch_size, steps_per_epoch)
    total_steps = steps_per_epoch * max(1, epochs)
    # Plain ASCII deliberately: this can run from a bare console, and on a
    # Windows codepage that is not UTF-8 (cp1250 here) an emoji in a print
    # raises UnicodeEncodeError and takes the whole run down with it.
    print(f"[train] YOLOX-{size}: {len(class_names)} class(es), {epochs} epochs, "
          f"{steps_per_epoch} steps/epoch on {device}")

    history = []
    train_seconds, train_images = 0.0, 0
    val_seconds, val_images = 0.0, 0
    best_val = float("inf")
    last_train_loss = float("nan")
    completed = 0
    os.makedirs(output_dir, exist_ok=True)
    weights_path = os.path.join(output_dir, f"yolox_{size}.pth")

    for epoch in range(1, epochs + 1):
        model.train()
        running, counted = 0.0, 0
        epoch_started = time.perf_counter()
        for step, (images, targets) in enumerate(train_loader, start=1):
            _check_stop()
            train_images += int(images.shape[0])
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            global_iter = (epoch - 1) * steps_per_epoch + (step - 1)
            lr = lr_scheduler.update_lr(global_iter + 1)
            for group in optimizer.param_groups:
                group["lr"] = lr

            outputs = model(images, targets)
            loss = outputs["total_loss"]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running += float(loss.detach().cpu())
            counted += 1
            done_steps = (epoch - 1) * steps_per_epoch + step
            if progress is not None:
                elapsed = time.perf_counter() - started
                progress(Progress(
                    epoch=epoch, total_epochs=epochs,
                    step=step, steps_per_epoch=steps_per_epoch,
                    loss=running / counted,
                    elapsed=elapsed,
                    eta=eta_seconds(done_steps, total_steps, elapsed),
                    stage="train",
                ))

        last_train_loss = running / max(1, counted)
        train_seconds += time.perf_counter() - epoch_started
        val_started = time.perf_counter()
        val_loss = _validate(model, val_loader, device, _check_stop)
        if val_loader is not None and len(val_loader):
            val_seconds += time.perf_counter() - val_started
            val_images += len(val_loader.dataset)
        history.append({"epoch": epoch, "train_loss": last_train_loss,
                        "val_loss": val_loss})
        completed = epoch
        print(f"   epoch {epoch}/{epochs}  train {last_train_loss:.4f}"
              + (f"  val {val_loss:.4f}" if val_loss == val_loss else ""))

        # Save on improvement, or unconditionally when there is nothing to
        # judge by — a run with no validation split should still leave a model.
        improved = val_loss < best_val if val_loss == val_loss else True
        if improved:
            best_val = val_loss if val_loss == val_loss else best_val
            torch.save({"model": model.state_dict(),
                        "class_names": class_names,
                        "size": size,
                        "image_size": list(image_size)}, weights_path)

        if on_epoch is not None:
            elapsed = time.perf_counter() - started
            done = epoch * steps_per_epoch
            try:
                on_epoch(EpochReport(
                    epoch=epoch, total_epochs=epochs,
                    train_loss=last_train_loss, val_loss=val_loss,
                    elapsed=elapsed,
                    eta=eta_seconds(done, total_steps, elapsed),
                    detector=_LiveDetector(model, device, class_names, image_size),
                ))
            except Cancelled:
                raise
            except Exception as e:  # noqa: BLE001 - a preview must never end a run
                print(f"[train] epoch report failed ({type(e).__name__}: {e})")

    labels_path = write_labels_sidecar(
        os.path.join(output_dir, "labels.json"), class_names)

    return TrainingResult(
        weights_path=weights_path,
        labels_path=labels_path,
        class_names=class_names,
        epochs_completed=completed,
        best_val_loss=best_val,
        final_train_loss=last_train_loss,
        seconds=time.perf_counter() - started,
        device=device,
        history=history,
        train_images_per_second=train_images / train_seconds if train_seconds > 0 else 0.0,
        val_images_per_second=val_images / val_seconds if val_seconds > 0 else 0.0,
        loop_seconds=train_seconds + val_seconds,
    )


def _validate(model, loader, device, check_stop) -> float:
    """Mean validation loss, or NaN when there is nothing to validate against.

    NaN rather than 0.0 or None so "not measured" cannot be mistaken for "a
    perfect score" by any comparison written later.

    An *empty* loader counts as nothing to validate against, not as zero loss.
    This was a live bug: a dataset whose validation split came out empty
    produced ``0.0``, which is better than any real model will ever score, so
    the best checkpoint was fixed at epoch one and every later improvement was
    discarded — while the log cheerfully printed ``val 0.0000`` each epoch.
    """
    if loader is None or len(loader) == 0:
        return float("nan")
    import torch

    model.train()   # YOLOX only computes losses in train mode
    total, batches = 0.0, 0
    with torch.no_grad():
        for images, targets in loader:
            check_stop()
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            total += float(model(images, targets)["total_loss"].detach().cpu())
            batches += 1
    return total / max(1, batches)
