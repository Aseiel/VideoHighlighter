"""The dataset loop as MCP tools: check a dataset, add a reviewed video by
the measured rule, ask whether it helped, and take out what did not.

The work is ``modules.teach.dataset_health``, ``modules.teach.intake`` and
``model_training.action_head.compare``; this file only decides what goes
back to the client.

**Class names stay on this machine unless the user says otherwise.** A
dataset's class names are the user's own vocabulary, and a tool result is
sent to the client's model. So by default every class is reported as a code
(``C01``, ``C02``, ...): the numbers a decision needs are all still there,
and the table that says which code is which class is a file beside the
dataset (``.intake/names/class_codes.json``) that the user reads. Setting
``mcp_class_names: true`` in config.yaml sends the real names instead. File
and video names are left out of coded results for the same reason.

The codes are made by the aliases the dataset reader already has: every
folder name is mapped to its classes' codes before anything is read, so no
code below this file ever sees a name it would have to hide.
"""
from __future__ import annotations

import contextlib
import json
import os
import threading
from typing import Optional, Sequence

from mcp_server.jobs import Job, JobCancelled, JobManager

NAMES_DIR = "names"
CODES_FILE = "class_codes.json"
SEEDS_MAX = 10


def _folder(path: str, what: str) -> str:
    path = os.path.abspath(os.path.expanduser(str(path)))
    if not os.path.isdir(path):
        raise ValueError(f"{what} not found: {path}")
    return path


_caller = threading.local()


@contextlib.contextmanager
def names_for(real: Optional[bool]):
    """Decide for the calls made inside it (on this thread) whether real class
    names are shown, whatever config.yaml says. The app's own chat uses it:
    a model served on this machine sends nothing anywhere, so there is nothing
    to hide from it. ``None`` leaves the config's answer."""
    before = getattr(_caller, "real", None)
    _caller.real = real
    try:
        yield
    finally:
        _caller.real = before


def sends_names() -> bool:
    """True when config.yaml says class names may go to the client."""
    if getattr(_caller, "real", None) is not None:
        return bool(_caller.real)
    from mcp_server.tools import get_config
    try:
        return bool(get_config()["config"].get("mcp_class_names", False))
    except Exception:  # noqa: BLE001 - an unreadable config means the safe default
        return False


class Names:
    """Folder names <-> what the client is shown, for one dataset."""

    def __init__(self, dataset: str, aliases_path: Optional[str] = None,
                 review: Optional[str] = None, real: Optional[bool] = None):
        from modules.teach import benchmark, intake

        self.dataset = dataset
        self.real = sends_names() if real is None else real
        self.user = benchmark.load_aliases(aliases_path)
        self.file = os.path.join(dataset, intake.MANIFEST_DIR, NAMES_DIR, CODES_FILE)
        self.aliases = self.user
        self.shown: dict = {}           # folder name -> what the client sees
        if self.real:
            return
        folders = set()
        for split in benchmark.SPLITS:
            where = os.path.join(dataset, split)
            if os.path.isdir(where):
                folders.update(n for n in os.listdir(where)
                               if os.path.isdir(os.path.join(where, n)))
        if review and os.path.isdir(review):
            folders.update(n for n in os.listdir(review)
                           if os.path.isdir(os.path.join(review, n)))
        codes = {}
        if os.path.isfile(self.file):
            with open(self.file, "r", encoding="utf-8") as handle:
                codes = json.load(handle)
        folders = {f for f in folders if not f.startswith(("_", "."))}
        labels_of = {f: benchmark.labels_of(f, self.user) for f in sorted(folders)}
        new = sorted({lb for labels in labels_of.values() for lb in labels} - set(codes))
        for label in new:
            codes[label] = f"C{len(codes) + 1:02d}"
        if new:
            os.makedirs(os.path.dirname(self.file), exist_ok=True)
            with open(self.file + ".tmp", "w", encoding="utf-8") as handle:
                json.dump(codes, handle, indent=1, ensure_ascii=False)
            os.replace(self.file + ".tmp", self.file)
        self.aliases = {f: benchmark.PAIR_SEPARATOR.join(codes[lb] for lb in labels)
                        for f, labels in labels_of.items()}
        self.shown = {f: (code or f"left-out-{i}")
                      for i, (f, code) in enumerate(sorted(self.aliases.items()))}

    def show(self, folder: str) -> str:
        return folder if self.real else self.shown.get(folder, "?")

    def folders(self, shown: Sequence[str]) -> list:
        """The real folders the client means by these names or codes."""
        if self.real:
            return list(shown or ())
        wanted = set(shown or ())
        return sorted(f for f, code in self.shown.items() if code in wanted)

    def note(self) -> dict:
        if self.real:
            return {"class_names": "real names (mcp_class_names is on in config.yaml)"}
        return {"class_names": "codes; the user can read which class each is in "
                               "class_codes_file. Set mcp_class_names: true in "
                               "config.yaml to send real names.",
                "class_codes_file": self.file}


# ── Quick ────────────────────────────────────────────────────────────────


def dataset_health(dataset: str, aliases: Optional[str] = None) -> dict:
    from modules.teach import dataset_health as DH

    dataset = _folder(dataset, "Dataset")
    names = Names(dataset, aliases)
    report = DH.check(dataset, names.aliases)
    if not names.real:
        for finding in report["findings"]:
            if finding["what"] not in ("few_videos", "one_video_floods_class"):
                finding["items"] = []       # file and video names
    return {**report, **names.note()}


def _plan(dataset, review, video, thin_under, cap, exclude, aliases):
    from modules.teach import intake

    dataset, review = _folder(dataset, "Dataset"), _folder(review, "Review folder")
    names = Names(dataset, aliases, review)
    planned = intake.plan(dataset, review, video, thin_under=thin_under, cap=cap,
                          aliases=names.aliases, exclude=names.folders(exclude))
    return names, planned


def _shown_plan(names: Names, planned: dict) -> dict:
    from modules.teach import intake

    out = intake.summary(planned)
    if not names.real:
        out.pop("review", None)
        for row in out["folders"]:
            row["folder"] = names.show(row["folder"])
    return {**out, **names.note()}


def plan_reviewed_video(dataset: str, review: str, video: str, thin_under: int = 200,
                        cap: int = 50, exclude: Sequence[str] = (),
                        aliases: Optional[str] = None) -> dict:
    names, planned = _plan(dataset, review, video, thin_under, cap, exclude, aliases)
    out = _shown_plan(names, planned)
    out["next"] = "add_reviewed_video with the same arguments adds these; nothing was changed"
    return out


def add_reviewed_video(dataset: str, review: str, video: str, thin_under: int = 200,
                       cap: int = 50, exclude: Sequence[str] = (),
                       aliases: Optional[str] = None) -> dict:
    from modules.teach import intake

    names, planned = _plan(dataset, review, video, thin_under, cap, exclude, aliases)
    out = _shown_plan(names, planned)
    out["applied"] = intake.apply(planned)
    out["next"] = ("start_dataset_comparison says whether this helped on the dataset's "
                   "other videos; undo_reviewed_video takes it all out again")
    return out


def drop_reviewed_folders(dataset: str, video: str, folders: Sequence[str],
                          aliases: Optional[str] = None) -> dict:
    from modules.teach import intake

    dataset = _folder(dataset, "Dataset")
    names = Names(dataset, aliases)
    real = names.folders(folders)
    if not real:
        raise ValueError("None of those folders is in this dataset.")
    out = intake.drop(dataset, video, real)
    if not names.real:
        out["by_folder"] = {names.show(f): n for f, n in out["by_folder"].items()}
        out["not_found"] = [names.show(f) for f in out["not_found"]]
    return out


def undo_reviewed_video(dataset: str, video: str) -> dict:
    from modules.teach import intake

    return intake.undo(_folder(dataset, "Dataset"), video)


# ── Jobs ─────────────────────────────────────────────────────────────────


def start_dataset_comparison(jobs: JobManager, dataset: str,
                             added: Optional[Sequence[str]] = None, seeds: int = 5,
                             parts: Optional[str] = "kinds",
                             aliases: Optional[str] = None) -> dict:
    dataset = _folder(dataset, "Dataset")
    if parts not in (None, "", "kinds", "folders"):
        raise ValueError('parts is "kinds", "folders" or null')
    seeds = max(2, min(int(seeds), SEEDS_MAX))
    names = Names(dataset, aliases)

    def run(job: Job) -> dict:
        from model_training.action_head import compare

        if names.real:
            aliases_file = aliases
        else:
            # The trainer reads aliases from a file; the coded ones live
            # beside the code table.
            aliases_file = os.path.join(os.path.dirname(names.file), "coded_aliases.json")
            with open(aliases_file, "w", encoding="utf-8") as handle:
                json.dump(names.aliases, handle, ensure_ascii=False)
        try:
            result = compare.compare(
                dataset, list(added) if added else None, seeds=seeds, parts=parts or None,
                aliases=aliases_file, log=job.log, should_stop=job.cancel_flag.is_set)
        except InterruptedError:
            raise JobCancelled() from None
        if "drop_folders" in result:
            result["drop_folders"] = sorted({names.show(f) for f in result["drop_folders"]})
            if parts == "folders" and not names.real:
                for row in result.get("parts", []):
                    row["part"] = names.show(row["part"])
                result["drop"] = list(result["drop_folders"])
                # The sentence was written with the folders' own names.
                head = result["summary"].split(" Parts that measured worse")[0]
                result["summary"] = head + (
                    " Parts that measured worse on their own: "
                    + ", ".join(result["drop"]) + "." if result["drop"] else "")
        result.pop("dataset", None)
        return {**result, **names.note()}

    job = jobs.submit("dataset_comparison", {
        "dataset": dataset, "added": list(added or []), "seeds": seeds, "parts": parts}, run)
    return {"job_id": job.id, "status": job.status}
