"""Train a round, score it on the held-out set, install it only if it wins.

Each round trains into ``runs/<n>/``, never over the model the app is using.
Then:

* **Scored on the frozen held-out set** (``build``), so rounds compare.
  Actions report *balanced accuracy* — the mean over classes of how often the
  model names a held-out clip correctly — and per class, which is the "how
  often does it find the thing" sentence docs/CUSTOM-MODEL-TRAINING.md asks
  for. Objects report the best validation loss, the measure
  ``train_yolox_run`` has.
* **Installed only when better** than the best round installed so far
  (``should_install``), or when asked. The previous model stays in its
  ``runs/`` folder, so going back is copying it again.

Actions train in a subprocess (``model_training.r3d.train``): minutes to hours
of torch that must not take the caller down with it, and whose log is kept in
``runs/<n>/train.log``. Objects train in-process through the same functions
the Training tab uses.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from typing import Callable, Optional

from modules.teach.build import DATASET_DIR
from modules.teach.project import ACTIONS, Project

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def next_round_dir(project: Project) -> tuple:
    number = len(project.rounds) + 1
    path = project.path("runs", f"{number:03d}")
    os.makedirs(path, exist_ok=True)
    return number, path


def is_better(candidate: dict, incumbent: Optional[dict]) -> bool:
    """Whether ``candidate``'s held-out score beats ``incumbent``'s."""
    if not incumbent:
        return True
    if "balanced_accuracy" in candidate:
        return candidate["balanced_accuracy"] > incumbent.get("balanced_accuracy", -1.0)
    if "best_val_loss" in candidate:
        return candidate["best_val_loss"] < incumbent.get("best_val_loss", float("inf"))
    return False


def installed_round(project: Project) -> Optional[dict]:
    for record in reversed(project.rounds):
        if record.get("installed"):
            return record
    return None


def actions_command(dataset: str, run_dir: str, epochs: int) -> list:
    return [sys.executable, "-m", "model_training.r3d.train",
            "--data-path", dataset,
            "--model-save-path", os.path.join(run_dir, "r3d_finetuned.pth"),
            "--checkpoint-dir", os.path.join(run_dir, "checkpoints"),
            "--metrics-out", os.path.join(run_dir, "metrics.json"),
            "--epochs", str(int(epochs)), "--no-viz"]


def train_actions(project: Project, run_dir: str, *, epochs: int,
                  run: Callable = subprocess.run) -> dict:
    dataset = project.path(DATASET_DIR)
    log_path = os.path.join(run_dir, "train.log")
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        result = run(actions_command(dataset, run_dir, epochs), cwd=_REPO,
                     stdout=log, stderr=subprocess.STDOUT)
    code = getattr(result, "returncode", 1)
    metrics_path = os.path.join(run_dir, "metrics.json")
    if code != 0 or not os.path.exists(metrics_path):
        raise RuntimeError(f"training failed (exit {code}); see {log_path}")
    with open(metrics_path, "r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    metrics["log"] = log_path
    return metrics


def train_objects(project: Project, run_dir: str, *, epochs: int,
                  progress: Optional[Callable] = None) -> dict:
    from training.train_yolox_run import train

    result = train(project.path(DATASET_DIR), run_dir, epochs=epochs,
                   progress=progress)
    return {"best_val_loss": float(result.best_val_loss),
            "epochs": int(result.epochs_completed),
            "device": result.device,
            "weights": result.weights_path,
            "labels": result.labels_path,
            "classes": list(result.class_names),
            "minutes": round(result.seconds / 60.0, 1)}


def install(project: Project, record: dict) -> dict:
    """Put a round's model where the app loads it from."""
    if project.task == ACTIONS:
        from modules.system.app_paths import import_r3d_action_model
        classes, variant = import_r3d_action_model(record["metrics"]["weights"],
                                                   record["metrics"]["mapping"])
        where = {"slot": "R3D action model", "classes": classes, "variant": variant}
    else:
        from modules.teach.project import slugify
        from training.export_yolox import install as install_detector
        result = install_detector(record["metrics"]["weights"],
                                  name=f"teach_{slugify(project.name)}")
        where = {"slot": "custom detector", "xml": result.xml_path,
                 "classes": list(result.class_names)}
    for other in project.rounds:
        other["installed"] = False
    record["installed"] = True
    record["installed_at"] = time.time()
    project.save()
    return where


def train_round(project: Project, *, epochs: Optional[int] = None,
                install_policy: str = "if-better",
                run: Callable = subprocess.run,
                progress: Optional[Callable] = None) -> dict:
    """Train on the built dataset; record the round; maybe install it.

    ``install_policy``: ``"if-better"`` (default), ``"always"``, ``"never"``.
    """
    if not os.path.isdir(project.path(DATASET_DIR)):
        raise RuntimeError("no dataset yet; run build first")
    epochs = int(epochs or project.settings.epochs)
    number, run_dir = next_round_dir(project)
    started = time.time()
    if project.task == ACTIONS:
        metrics = train_actions(project, run_dir, epochs=epochs, run=run)
    else:
        metrics = train_objects(project, run_dir, epochs=epochs, progress=progress)

    from modules.teach.build import built_signature

    record = {"round": number, "dir": run_dir, "started": started,
              "dataset": built_signature(project),
              "seconds": round(time.time() - started, 1), "epochs": epochs,
              "accepted": {c: len(project.accepted(c)) for c in project.class_names()},
              "metrics": metrics, "installed": False}
    incumbent = installed_round(project)
    better = is_better(metrics, incumbent["metrics"] if incumbent else None)
    record["better_than_installed"] = better
    project.rounds.append(record)
    project.save()

    installed = None
    if install_policy == "always" or (install_policy == "if-better" and better):
        installed = install(project, record)
    return {"round": number, "metrics": metrics, "better_than_installed": better,
            "installed": installed,
            "kept_previous": None if installed or not incumbent else incumbent["round"]}
