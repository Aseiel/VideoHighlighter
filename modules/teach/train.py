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

Actions train in a subprocess (``model_training.action_head.train``, a head on
the SigLIP2 frame encoder): minutes on most machines
of torch that must not take the caller down with it, and whose log is kept in
``runs/<n>/train.log``. Objects train in-process through the same functions
the Training tab uses.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from typing import Callable, Optional

from modules.teach.build import DATASET_DIR
from modules.teach.project import ACTIONS, TRAIN, VAL, Project

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


def installed_detector_files(project: Project) -> Optional[dict]:
    """``{"round", "xml", "classes"}`` of the installed detector, or None.

    Rounds record where they were installed. A round installed before that
    was recorded is found where ``install`` puts every project's detector.
    """
    record = installed_round(project)
    if record is None or project.task == ACTIONS:
        return None
    where = record.get("install") or {}
    xml, classes = where.get("xml"), where.get("classes")
    if not xml:
        from modules.teach.project import slugify
        from training.export_yolox import DEFAULT_DEST

        name = f"teach_{slugify(project.name)}"
        xml = os.path.join(os.path.abspath(DEFAULT_DEST), name, f"{name}.xml")
    if not classes:
        try:
            with open(os.path.join(os.path.dirname(xml), "labels.json"),
                      "r", encoding="utf-8") as handle:
                classes = list(json.load(handle))
        except (OSError, ValueError):
            classes = []
    return {"round": record.get("round"), "xml": xml, "classes": classes}


HEAD_DIR = "head"


def actions_command(dataset: str, run_dir: str, sources: int) -> list:
    """The head trainer on the built dataset. It picks its own training length
    from out-of-fold scores, so a round has no epochs to pass. Trust needs
    held-out hits from several source videos (three when there are); a
    one-video project can only ask for one."""
    return [sys.executable, "-m", "model_training.action_head.train",
            *actions_args(dataset, run_dir, sources)]


def actions_args(dataset: str, run_dir: str, sources: int) -> list:
    """The trainer's arguments, without the interpreter (see actions_command)."""
    return ["--data-path", dataset,
            "--out", os.path.join(run_dir, HEAD_DIR),
            "--min-videos", str(max(1, min(3, int(sources))))]


def _sources(dataset: str) -> int:
    """How many source videos the built dataset holds (names before _temp)."""
    from modules.teach.benchmark import group_of

    names = set()
    for split in (TRAIN, VAL):
        for root, _dirs, files in os.walk(os.path.join(dataset, split)):
            names.update(group_of(f) for f in files if "_temp" in f)
    return len(names) or 1


def _train_in_process(args: list, log) -> int:
    """The head trainer in this process, every line into ``log``.

    Not ``sys.executable -m ...``: in the packaged app ``sys.executable`` is
    the app itself, and a child started that way opened a second copy of the
    app instead of training."""
    def write(text):
        log.write(f"{text}\n")
        log.flush()
    try:
        from model_training.action_head import train as trainer
        return trainer.main(args, log=write)
    except Exception:                                  # noqa: BLE001 - into the log
        import traceback
        log.write(traceback.format_exc())
        return 1


def train_actions(project: Project, run_dir: str, *, epochs: int,
                  run: Optional[Callable] = None) -> dict:
    """``run`` (a ``subprocess.run`` stand-in) runs the trainer as a command;
    left out, it runs in this process, which also works in the packaged app."""
    dataset = project.path(DATASET_DIR)
    log_path = os.path.join(run_dir, "train.log")
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        if run is None:
            code = _train_in_process(actions_args(dataset, run_dir, _sources(dataset)), log)
        else:
            result = run(actions_command(dataset, run_dir, _sources(dataset)), cwd=_REPO,
                         stdout=log, stderr=subprocess.STDOUT)
            code = getattr(result, "returncode", 1)
    head = os.path.join(run_dir, HEAD_DIR)
    meta_path = os.path.join(head, "head.json")
    if code != 0 or not os.path.exists(meta_path):
        raise RuntimeError(f"training failed (exit {code}); see {log_path}")
    with open(meta_path, "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    single = (meta.get("heldout") or {}).get("single") or {}
    return {"balanced_accuracy": float(single.get("balanced_accuracy", 0.0)),
            "accuracy": float(single.get("accuracy", 0.0)),
            "classes": list(meta.get("classes", [])),
            "trusted": sum(t is not None for t in meta.get("trust_thresholds", [])),
            "head": head, "log": log_path}


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
        from modules.system.app_paths import action_models_dir
        from modules.teach.project import slugify
        dest = os.path.join(action_models_dir(), f"teach-{slugify(project.name)}")
        if os.path.isdir(dest):
            shutil.rmtree(dest)
        shutil.copytree(record["metrics"]["head"], dest)
        where = {"slot": "action model", "head": dest,
                 "classes": list(record["metrics"].get("classes", []))}
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
    # Where it went, so the next round's sort and box proposals can use it.
    record["install"] = where
    project.save()
    return where


def train_round(project: Project, *, epochs: Optional[int] = None,
                install_policy: str = "if-better",
                run: Optional[Callable] = None,
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
    if project.task != ACTIONS:
        # What this round learnt from, for sharing it later: counted now,
        # because by then the project may have grown past it.
        from modules.teach.share import measured
        record.update(measured(project))
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
