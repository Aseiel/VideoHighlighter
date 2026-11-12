"""The dataset loop for the chat assistant: check, add, measure, take out, train.

Claude Desktop reaches this loop through the MCP server (``mcp_server/``). A
model served by LM Studio on this machine is reached through the app's own
chat (``llm/assistant_agent.py``), and until now that chat had four tools and
none of them touched a dataset. These are the same operations, the same code
underneath (``mcp_server.dataset_tools``), shaped for a small local model:

* **No jobs to follow.** Over MCP a long operation is ``start_*`` plus
  ``get_job`` until done, because a tool call there must answer in seconds. A
  small model polls badly: it spends its steps asking, or stops asking. Here a
  tool waits and returns the result, and what the job logs goes to the chat
  as progress, never into the model's context.
* **The rule is in the tool, not in the prompt.** "Keep an addition only when
  it did not measure worse" is a procedure: add, compare, drop the folders
  that measured worse, compare again if the whole was worse, take it all out
  if it still is. ``improve_dataset`` does exactly that, so the outcome does
  not depend on a 4 GB model carrying a five-step plan. The model's part is
  choosing what to run and telling the user what happened.
* **A trained model replaces the installed one only when it is not worse** on
  held-out videos (``train_action_model``). The candidate is trained beside
  the installed model, never over it, and the one it replaces is kept.

Class names: a model on this machine is shown the real ones (``local=True``).
Anything else gets what ``mcp_class_names`` in config.yaml allows, as over MCP.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from typing import Callable, Optional

from llm.assistant_agent import Cancelled, Tool

POLL_SECONDS = 1.0
SAY_EVERY = 4.0             # seconds between progress lines in the chat
# Held-out accuracy moves about a point between runs on the same clips
# (compare.OVERALL_MIN); a candidate inside that is not worse.
INSTALL_MARGIN = 0.01
IF_BETTER, ALWAYS, NEVER = "if_better", "always", "never"

PROMPT = """\

You can also keep a training dataset healthy and train the action model from \
it. A dataset is a folder with train/, val/ and test/, one sub-folder per \
class. A reviewed video is a folder with one sub-folder per class holding that \
video's checked crops.
- dataset_health first: it says which classes are thin and what to fix.
- improve_dataset adds a reviewed video and decides by measurement: it keeps \
only what did not measure worse and takes the rest out again. It takes 5-20 \
minutes; say so before calling it. Report its verdict and numbers as given.
- train_action_model trains on the dataset and replaces the installed model \
only when the new one is not worse on held-out videos. It also takes minutes.
- Never claim an addition or a model is better unless the tool's result says \
so. Ask the user for the dataset and review folder paths if you do not have \
them."""

LABELS = {"dataset_health": "checking the dataset",
          "plan_reviewed_video": "planning what the video would add",
          "improve_dataset": "adding the video and measuring it",
          "measure_dataset": "measuring what was added",
          "undo_reviewed_video": "taking the video out again",
          "train_action_model": "training the action model"}


class _Say:
    """Progress into the chat, at most one line every few seconds."""

    def __init__(self, say: Callable[[str], None], every: float = SAY_EVERY):
        self.say, self.every, self.last = say, every, 0.0

    def __call__(self, line) -> None:
        text = str(line).strip()
        now = time.monotonic()
        if text and now - self.last >= self.every:
            self.last = now
            self.say(text.splitlines()[-1][:200])


def _wait(jobs, started: dict, cancelled: Callable[[], bool], say: Callable) -> dict:
    """Follow a job to its end, so the model does not have to."""
    from mcp_server import jobs as J

    job = jobs.get(started["job_id"])
    shown = ""
    while True:
        done = job.status in J.FINISHED
        tail = job.snapshot(1)["log_tail"]
        if tail and tail[-1] != shown:
            shown = tail[-1]
            say(shown)
        if done:
            break
        if cancelled():
            jobs.cancel(job.id)     # stops at its next check; the chat does not wait
            raise Cancelled()
        time.sleep(POLL_SECONDS)
    if job.status == J.FAILED:
        # The job's own error often points at "the lines above"; the model
        # only gets this text, so the lines come with it.
        tail = " | ".join(line.strip() for line in job.snapshot(3)["log_tail"]
                          if line.strip())
        raise RuntimeError((job.error or "the job failed") + (f" ({tail})" if tail else ""))
    if job.status == J.CANCELLED:
        raise Cancelled()
    return job.result


def _short(result: dict) -> dict:
    """A comparison without its longest tables: what a small model can hold."""
    out = {k: result[k] for k in (
        "verdict", "keep", "summary", "top1_change", "top1_without", "top1_with",
        "classes_better", "classes_worse", "drop_folders", "only_in_added", "clips")
        if k in result}
    rows = result.get("per_class") or []
    if rows:
        out["largest_class_changes"] = [
            {k: r[k] for k in ("class", "recall_without", "recall_with", "change", "verdict")}
            for r in sorted(rows, key=lambda r: -abs(r["change"]))[:8]]
    return out


def _read_meta(folder: str) -> Optional[dict]:
    try:
        with open(os.path.join(folder, "head.json"), encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _accuracy(meta: Optional[dict]) -> Optional[float]:
    value = (((meta or {}).get("heldout") or {}).get("single") or {}).get("accuracy")
    return None if value is None else float(value)


def decide_install(new: dict, old: Optional[dict], install: str = IF_BETTER) -> tuple:
    """``(install?, why)`` for a freshly trained head against the installed one."""
    if install == NEVER:
        return False, "install was 'never': the candidate is kept beside the installed model"
    if install == ALWAYS:
        return True, "install was 'always'"
    if old is None:
        return True, "no model of that name was installed"
    a_new, a_old = _accuracy(new), _accuracy(old)
    if a_new is None:
        return False, "the new model has no held-out score to judge it by"
    if a_old is None:
        return True, "the installed model has no held-out score; the new one does"
    if sorted(new.get("classes") or []) != sorted(old.get("classes") or []):
        return False, (
            f"the classes changed ({len(old.get('classes') or [])} installed, "
            f"{len(new.get('classes') or [])} new), so the two scores "
            f"({a_old:.3f} and {a_new:.3f}) are not on the same question. Ask the "
            f"user; install='always' installs it")
    if a_new + INSTALL_MARGIN >= a_old:
        return True, (f"held-out accuracy {a_new:.3f} against the installed "
                      f"{a_old:.3f}: not worse")
    return False, (f"held-out accuracy {a_new:.3f} is below the installed "
                   f"{a_old:.3f}; the installed model stays")


def dataset_tools(*, cancelled: Callable[[], bool] = lambda: False,
                  say: Callable[[str], None] = lambda _line: None,
                  local: bool = False, jobs=None) -> list:
    """The dataset tools. ``say`` gets progress lines for the chat; ``local``
    is whether the model runs on this machine (real class names are shown)."""
    from mcp_server import dataset_tools as DT
    from mcp_server.jobs import JobManager

    jobs = jobs if jobs is not None else JobManager()
    real = True if local else None

    def _common(args) -> dict:
        return {"dataset": str(args.get("dataset") or ""),
                "aliases": args.get("aliases") or None}

    def _video_args(args) -> dict:
        out = {**_common(args), "review": str(args.get("review") or ""),
               "video": str(args.get("video") or "")}
        if not out["video"]:
            raise ValueError("video is the name this video gets in the dataset")
        for key in ("thin_under", "cap"):
            if args.get(key) is not None:
                out[key] = int(args[key])
        if args.get("exclude"):
            out["exclude"] = list(args["exclude"])
        return out

    def _compare(dataset, added, seeds, aliases) -> dict:
        progress = _Say(say)
        started = DT.start_dataset_comparison(jobs, dataset, added=added, seeds=seeds,
                                              parts="kinds", aliases=aliases)
        return _wait(jobs, started, cancelled, progress)

    def health(args):
        with DT.names_for(real):
            report = DT.dataset_health(**_common(args))
        # The class table is the long part; the findings and what is wanted say it.
        report.pop("classes_table", None)
        return report

    def plan(args):
        with DT.names_for(real):
            return DT.plan_reviewed_video(**_video_args(args))

    def improve(args):
        a = _video_args(args)
        seeds = int(args.get("seeds") or 5)
        with DT.names_for(real):
            added = DT.add_reviewed_video(**a)
            applied = added.get("applied") or {}
            if not applied.get("in_dataset"):
                return {"outcome": "nothing added",
                        "why": "no folder of this video belongs to a thin class; "
                               "plan_reviewed_video says why for each",
                        "folders": added.get("folders")}
            dataset, video = a["dataset"], a["video"]
            steps = [f"added {applied['in_dataset']} clips"]
            try:
                result = _compare(dataset, [video], seeds, a["aliases"])
                dropped = result.get("drop_folders") or []
                if dropped:
                    out = DT.drop_reviewed_folders(dataset, video, dropped,
                                                   aliases=a["aliases"])
                    steps.append(f"took out {out['removed']} clips of folders that "
                                 f"measured worse on their own: {', '.join(dropped)}")
                    if not out["in_dataset"]:
                        DT.undo_reviewed_video(dataset, video)
                        return {"outcome": "taken out", "steps": steps,
                                "why": "every folder it added measured worse",
                                "measured": _short(result)}
                    if not result["keep"]:
                        steps.append("the whole was worse, so what is left was "
                                     "measured again")
                        result = _compare(dataset, [video], seeds, a["aliases"])
                if not result["keep"]:
                    gone = DT.undo_reviewed_video(dataset, video)
                    steps.append(f"took out all {gone['removed']} clips")
                    return {"outcome": "taken out", "steps": steps,
                            "why": "it measured worse on the dataset's other videos",
                            "measured": _short(result)}
            except BaseException:
                # Stopped or failed before it was measured: an addition nobody
                # measured is not left in the dataset.
                try:
                    DT.undo_reviewed_video(dataset, video)
                except Exception as e:  # noqa: BLE001 - the first error is the one to report
                    print(f"⚠️ dataset agent: could not take {video} out again: {e}")
                raise
            return {"outcome": "kept", "steps": steps, "measured": _short(result),
                    "next": "train_action_model trains on the dataset as it is now"}

    def measure(args):
        common = _common(args)
        with DT.names_for(real):
            result = _compare(common["dataset"], args.get("added") or None,
                              int(args.get("seeds") or 5), common["aliases"])
        return _short(result)

    def undo(args):
        return DT.undo_reviewed_video(str(args.get("dataset") or ""),
                                      str(args.get("video") or ""))

    def train(args):
        from model_training.action_head import train as trainer
        from modules.system import app_paths

        dataset = DT._folder(args.get("dataset") or "", "Dataset")
        name = str(args.get("name") or trainer.DEFAULT_NAME)
        if not name.replace("-", "").replace("_", "").isalnum():
            raise ValueError("name is letters, digits, - and _ only")
        install = str(args.get("install") or IF_BETTER)
        if install not in (IF_BETTER, ALWAYS, NEVER):
            raise ValueError(f"install is {IF_BETTER}, {ALWAYS} or {NEVER}")
        installed = trainer.default_out(name)
        work = os.path.join(app_paths.user_data_dir(), "cache", "action_head")
        candidate = os.path.join(work, "candidates", name)
        shutil.rmtree(candidate, ignore_errors=True)

        lines: list = []
        progress = _Say(say)

        def log(line):
            lines.append(str(line))
            del lines[:-40]
            print(line)
            progress(line)

        argv = ["--data-path", dataset, "--name", name, "--out", candidate]
        if args.get("aliases"):
            argv += ["--aliases", str(args["aliases"])]
        code = trainer.main(argv, log=log, should_stop=cancelled)
        if code == trainer.STOPPED:
            raise Cancelled()
        new = _read_meta(candidate)
        if code != 0 or new is None:
            raise RuntimeError("Training did not finish: "
                               + " | ".join(x.strip() for x in lines[-4:] if x.strip()))
        old = _read_meta(installed)
        go, why = decide_install(new, old, install)
        out = {"trained": True, "installed": go, "why": why, "name": name,
               "classes": len(new.get("classes") or []),
               "heldout_accuracy": _accuracy(new),
               "installed_before_accuracy": _accuracy(old),
               "heldout": {k: (new.get("heldout") or {}).get(k) for k in ("clips", "videos")}}
        with DT.names_for(real):
            if DT.sends_names():
                rows = [(n, c) for n, c in (new.get("per_class") or {}).items()
                        if c.get("heldout_recall") is not None]
                out["weakest_classes"] = [
                    {"class": n, "heldout_recall": c["heldout_recall"],
                     "clips": c["clips"], "videos": c["videos"]}
                    for n, c in sorted(rows, key=lambda r: r[1]["heldout_recall"])[:6]]
        if go:
            if os.path.isdir(installed):
                previous = os.path.join(work, "previous", name)
                shutil.rmtree(previous, ignore_errors=True)
                os.makedirs(os.path.dirname(previous), exist_ok=True)
                shutil.move(installed, previous)
                out["previous_model_kept_at"] = previous
            os.makedirs(os.path.dirname(installed), exist_ok=True)
            shutil.move(candidate, installed)
            out["model"] = installed
        else:
            out["candidate_kept_at"] = candidate
        return out

    text = {"type": "string"}
    dataset_arg = {"dataset": {**text, "description":
                               "Full path of the dataset folder (holds train/, val/, test/)."},
                   "aliases": {**text, "description":
                               "Optional JSON file renaming or leaving out class folders."}}
    video_args = {**dataset_arg,
                  "review": {**text, "description":
                             "Folder with one sub-folder per class holding the "
                             "video's checked crops."},
                  "video": {**text, "description":
                            "Short name this video gets in the dataset."},
                  "thin_under": {"type": "integer", "description":
                                 "A class is thin below this many clips (default 200)."},
                  "cap": {"type": "integer", "description":
                          "Most crops taken per folder (default 50)."},
                  "exclude": {"type": "array", "items": text,
                              "description": "Folders to leave out."}}
    seeds = {"seeds": {"type": "integer", "description":
                       "Training repeats the verdict rests on (default 5)."}}

    def obj(properties, required):
        return {"type": "object", "properties": properties, "required": required}

    return [
        Tool("dataset_health",
             "Is a training dataset healthy? Reads file names only (seconds). "
             "Returns findings, each with what is wrong and the fix, and "
             "'wanted': the classes more videos would help most.",
             obj(dataset_arg, ["dataset"]), health),
        Tool("plan_reviewed_video",
             "What adding a reviewed video to the dataset would do. Changes "
             "nothing. Per folder: what was found, what would be taken, why not.",
             obj(video_args, ["dataset", "review", "video"]), plan),
        Tool("improve_dataset",
             "Add a reviewed video to the dataset and keep only what helps: adds "
             "its crops (thin classes only, capped), measures on the dataset's "
             "other videos with and without them, takes out the folders that "
             "measured worse, and takes out everything if the result is still "
             "worse. Waits until done: 5-20 minutes. Returns outcome ('kept', "
             "'taken out' or 'nothing added'), the steps taken and the numbers.",
             obj({**video_args, **seeds}, ["dataset", "review", "video"]), improve),
        Tool("measure_dataset",
             "Measure whether the videos already added to the dataset help on "
             "its other videos. Changes nothing. Waits until done: 5-20 minutes. "
             "added: video names (default: every video improve_dataset added).",
             obj({**dataset_arg, **seeds,
                  "added": {"type": "array", "items": text}}, ["dataset"]), measure),
        Tool("undo_reviewed_video",
             "Take out of the dataset everything a reviewed video added: those "
             "files and only those.",
             obj({"dataset": dataset_arg["dataset"], "video": video_args["video"]},
                 ["dataset", "video"]), undo),
        Tool("train_action_model",
             "Train the action model on the dataset and score it on videos it "
             "did not train on. Waits until done: minutes. install: 'if_better' "
             "(default) replaces the installed model of that name only when the "
             "new one is not worse on held-out videos, 'always', or 'never'. The "
             "replaced model is kept. Returns both accuracies and what was done.",
             obj({**dataset_arg,
                  "name": {**text, "description": "Model name (default taught-actions)."},
                  "install": {**text, "enum": [IF_BETTER, ALWAYS, NEVER]}},
                 ["dataset"]), train),
    ]
