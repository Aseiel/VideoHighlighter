"""The dataset loop as tools of the app's own chat (`llm/dataset_agent.py`).

For a model served on this machine: no job to poll, the keep-or-take-out rule
inside the tool, and a trained model installed only when it is not worse. The
comparison and the trainer are faked; what is pinned is what the tools do with
their answers, and that nothing unmeasured is ever left in the dataset.
"""
from __future__ import annotations

import json
import os
import threading

import pytest

pytest.importorskip("mcp")

from llm import assistant_agent as aa
from llm import dataset_agent as DA
from mcp_server import dataset_tools as DT
from mcp_server.jobs import JobManager
from modules.teach import intake


def _clip(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(os.path.basename(path).encode())


@pytest.fixture
def folders(tmp_path):
    dataset, review = str(tmp_path / "dataset"), str(tmp_path / "review")
    for i in range(12):
        _clip(os.path.join(dataset, "train", "big", f"film{i % 4}_temp_clip_{i}.mp4"))
    for i in range(3):
        _clip(os.path.join(dataset, "train", "thin", f"film{i}_temp_clip_t{i}.mp4"))
    for i in range(20):
        _clip(os.path.join(review, "thin", f"src__{i:08d}_crop.mp4"))
    for i in range(4):
        _clip(os.path.join(review, "fresh", f"src__{i:08d}_crop.mp4"))
    return dataset, review


def _measured(keep=True, verdict="no clear change", drop=()):
    return {"verdict": verdict, "keep": keep, "summary": f"{verdict}.",
            "top1_change": 0.0 if keep else -0.04, "drop_folders": list(drop),
            "per_class": [{"class": "thin", "recall_without": 0.5, "recall_with": 0.6,
                           "change": 0.1, "verdict": "better", "dataset_clips": 3}]}


@pytest.fixture
def comparisons(monkeypatch):
    """Scripted answers for start_dataset_comparison; records each call."""
    script, calls = [], []

    def start(jobs, dataset, added=None, seeds=5, parts="kinds", aliases=None):
        calls.append({"added": added, "seeds": seeds,
                      "in_dataset": len((intake.read_manifest(dataset, added[0]) or {})
                                        .get("added", [])) if added else None})
        answer = script.pop(0)

        def run(job):
            job.log("fold 1 of 5")
            if callable(answer):
                return answer(job)
            return answer

        job = jobs.submit("dataset_comparison", {}, run)
        return {"job_id": job.id, "status": job.status}

    monkeypatch.setattr(DT, "start_dataset_comparison", start)
    monkeypatch.setattr(DA, "POLL_SECONDS", 0.01)
    return script, calls


def _tools(local=True, cancelled=lambda: False, say=lambda _l: None):
    return {t.name: t for t in DA.dataset_tools(cancelled=cancelled, say=say, local=local,
                                                jobs=JobManager())}


def _args(dataset, review):
    return {"dataset": dataset, "review": review, "video": "newfilm",
            "thin_under": 10, "cap": 6}


def _added(dataset):
    return len((intake.read_manifest(dataset, "newfilm") or {}).get("added", []))


def test_the_tools_are_few_and_none_asks_the_model_to_follow_a_job():
    tools = _tools()
    assert list(tools) == ["dataset_health", "plan_reviewed_video", "improve_dataset",
                           "measure_dataset", "undo_reviewed_video", "train_action_model"]
    assert not any("job" in t.description.lower() or t.name.startswith("start_")
                   for t in tools.values())
    for t in tools.values():
        assert t.schema["type"] == "object" and set(t.schema["required"]) <= set(
            t.schema["properties"])
        assert t.name in DA.LABELS


def test_what_did_not_measure_worse_is_kept(folders, comparisons):
    dataset, review = folders
    script, calls = comparisons
    script.append(_measured(keep=True))
    said = []
    out = _tools(say=said.append)["improve_dataset"].fn(_args(dataset, review))
    assert out["outcome"] == "kept" and out["measured"]["keep"] is True
    assert _added(dataset) == 10                       # 6 thin + 4 fresh
    assert calls == [{"added": ["newfilm"], "seeds": 5, "in_dataset": 10}]
    assert said == ["fold 1 of 5"]                     # progress went to the chat
    assert "per_class" not in out["measured"]          # the long table stays out


def test_what_measured_worse_is_taken_out_again(folders, comparisons):
    dataset, review = folders
    comparisons[0].append(_measured(keep=False, verdict="worse"))
    out = _tools()["improve_dataset"].fn(_args(dataset, review))
    assert out["outcome"] == "taken out" and "worse" in out["why"]
    assert intake.read_manifest(dataset, "newfilm") is None
    assert len(os.listdir(os.path.join(dataset, "train", "thin"))) == 3


def test_only_the_folders_that_measured_worse_go(folders, comparisons):
    dataset, review = folders
    script, calls = comparisons
    script.append(_measured(keep=True, drop=["fresh"]))
    out = _tools()["improve_dataset"].fn(_args(dataset, review))
    assert out["outcome"] == "kept" and len(calls) == 1
    assert _added(dataset) == 6 and "fresh" in out["steps"][1]


def test_a_worse_whole_is_measured_again_after_its_bad_folders_go(folders, comparisons):
    dataset, review = folders
    script, calls = comparisons
    script += [_measured(keep=False, verdict="worse", drop=["fresh"]),
               _measured(keep=False, verdict="worse")]
    out = _tools()["improve_dataset"].fn(_args(dataset, review))
    assert [c["in_dataset"] for c in calls] == [10, 6]
    assert out["outcome"] == "taken out" and intake.read_manifest(dataset, "newfilm") is None

    script += [_measured(keep=False, verdict="worse", drop=["fresh"]), _measured(keep=True)]
    out = _tools()["improve_dataset"].fn(_args(dataset, review))
    assert out["outcome"] == "kept" and _added(dataset) == 6


def test_stopping_leaves_nothing_unmeasured_in_the_dataset(folders, comparisons):
    dataset, review = folders
    stop, running = threading.Event(), threading.Event()

    def slow(job):
        running.set()
        while not job.cancel_flag.is_set():
            job.cancel_flag.wait(0.01)
        return {}

    comparisons[0].append(slow)

    def cancelled():
        if running.is_set():
            stop.set()
        return stop.is_set()

    with pytest.raises(aa.Cancelled):
        _tools(cancelled=cancelled)["improve_dataset"].fn(_args(dataset, review))
    assert intake.read_manifest(dataset, "newfilm") is None


def test_a_failed_measurement_is_an_error_and_the_addition_is_undone(folders, comparisons):
    dataset, review = folders

    def boom(job):
        raise ValueError("no encoder")

    comparisons[0].append(boom)
    with pytest.raises(RuntimeError, match="no encoder"):
        _tools()["improve_dataset"].fn(_args(dataset, review))
    assert intake.read_manifest(dataset, "newfilm") is None


def test_a_video_with_nothing_for_a_thin_class_adds_nothing(folders, comparisons):
    dataset, review = folders
    out = _tools()["improve_dataset"].fn({**_args(dataset, review), "thin_under": 1,
                                          "exclude": ["fresh"]})
    assert out["outcome"] == "nothing added" and comparisons[1] == []


def test_a_model_on_this_machine_sees_names_and_any_other_the_config(folders, monkeypatch):
    dataset, review = folders
    monkeypatch.setattr("mcp_server.tools.get_config",
                        lambda: {"config": {"mcp_class_names": False}})
    here = _tools(local=True)["plan_reviewed_video"].fn(_args(dataset, review))
    assert {r["folder"] for r in here["folders"]} == {"thin", "fresh"}
    away = _tools(local=False)["plan_reviewed_video"].fn(_args(dataset, review))
    assert not {"thin", "fresh"} & {r["folder"] for r in away["folders"]}
    assert not DT.sends_names()                        # the override ended with the call
    health = _tools(local=True)["dataset_health"].fn({"dataset": dataset})
    assert "classes_table" not in health and health["findings"]


def test_only_lm_studio_on_this_machine_counts_as_here():
    assert aa.runs_here(aa.OpenAIToolTurn("http://localhost:1234", "m").converse)
    assert aa.runs_here(aa.OpenAIToolTurn("http://127.0.0.1:1234", "m").converse)
    assert not aa.runs_here(aa.OpenAIToolTurn("http://192.168.1.20:1234", "m").converse)
    assert not aa.runs_here(lambda *a, **k: None)      # Claude's converse


# ── training ─────────────────────────────────────────────────────────────


def _meta(accuracy, classes=("a", "b")):
    return {"classes": list(classes), "heldout": {"clips": 40, "videos": 8,
                                                  "single": {"accuracy": accuracy}},
            "per_class": {c: {"clips": 20, "videos": 4, "heldout_recall": 0.5 + i / 10}
                          for i, c in enumerate(classes)}}


def test_a_new_model_is_installed_only_when_it_is_not_worse():
    assert DA.decide_install(_meta(0.6), None)[0]
    assert DA.decide_install(_meta(0.62), _meta(0.6))[0]
    assert DA.decide_install(_meta(0.595), _meta(0.6))[0]          # inside run-to-run noise
    go, why = DA.decide_install(_meta(0.5), _meta(0.6))
    assert not go and "0.500" in why and "0.600" in why
    # More classes is a different question: not decided for the user.
    go, why = DA.decide_install(_meta(0.7, ("a", "b", "c")), _meta(0.6))
    assert not go and "always" in why
    assert DA.decide_install(_meta(0.1), _meta(0.6), DA.ALWAYS)[0]
    assert not DA.decide_install(_meta(0.9), _meta(0.6), DA.NEVER)[0]


@pytest.fixture
def trainer(tmp_path, monkeypatch):
    from model_training.action_head import train
    from modules.system import app_paths

    monkeypatch.setattr(app_paths, "user_data_dir", lambda: str(tmp_path / "data"))
    runs = {"accuracy": 0.6, "code": 0, "argv": None}

    def main(argv, *, log=print, should_stop=None):
        runs["argv"] = list(argv)
        if runs["code"]:
            log("❌ Too few clips to train")
            return runs["code"]
        out = argv[argv.index("--out") + 1]
        os.makedirs(out)
        log("Training the saved head")
        with open(os.path.join(out, "head.json"), "w", encoding="utf-8") as fh:
            json.dump(_meta(runs["accuracy"]), fh)
        with open(os.path.join(out, "head.onnx"), "wb") as fh:
            fh.write(str(runs["accuracy"]).encode())
        return 0

    monkeypatch.setattr(train, "main", main)
    return runs, train.default_out(train.DEFAULT_NAME)


def _installed_accuracy(folder):
    with open(os.path.join(folder, "head.onnx"), "rb") as fh:
        return float(fh.read())


def test_training_never_overwrites_a_better_installed_model(folders, trainer):
    dataset, _review = folders
    runs, installed = trainer
    train = _tools()["train_action_model"].fn

    first = train({"dataset": dataset})
    assert first["installed"] and first["model"] == installed
    assert "--out" in runs["argv"] and installed not in runs["argv"]   # trained beside it
    assert [r["class"] for r in first["weakest_classes"]] == ["a", "b"]

    runs["accuracy"] = 0.4
    worse = train({"dataset": dataset})
    assert not worse["installed"] and worse["installed_before_accuracy"] == 0.6
    assert _installed_accuracy(installed) == 0.6
    assert os.path.isfile(os.path.join(worse["candidate_kept_at"], "head.json"))

    runs["accuracy"] = 0.7
    better = train({"dataset": dataset})
    assert better["installed"] and _installed_accuracy(installed) == 0.7
    assert _installed_accuracy(better["previous_model_kept_at"]) == 0.6

    runs["code"] = 1
    with pytest.raises(RuntimeError, match="Too few clips"):
        train({"dataset": dataset})
    assert _installed_accuracy(installed) == 0.7
    with pytest.raises(ValueError):
        train({"dataset": dataset, "name": "../elsewhere"})


def test_class_names_from_training_follow_the_same_rule(folders, trainer, monkeypatch):
    monkeypatch.setattr("mcp_server.tools.get_config", lambda: {"config": {}})
    out = _tools(local=False)["train_action_model"].fn({"dataset": folders[0]})
    assert out["installed"] and "weakest_classes" not in out


# ── in the chat's loop ───────────────────────────────────────────────────


def test_one_call_does_the_whole_loop_for_a_model(folders, comparisons):
    dataset, review = folders
    comparisons[0].append(_measured(keep=False, verdict="worse"))
    call = {"type": "tool_use", "id": "t1", "name": "improve_dataset",
            "input": _args(dataset, review)}
    replies = [[call], [{"type": "text", "text": "It measured worse, so I took it out."}]]
    sent = []

    def model(messages, **kw):
        sent.append(json.loads(json.dumps(messages)))
        content = replies.pop(0)
        return {"content": content,
                "stop_reason": "tool_use" if content[0]["type"] == "tool_use" else "end_turn"}

    answer, used = aa.run(model, aa.SYSTEM_PROMPT + DA.PROMPT, "add the review",
                          list(_tools().values()), call_on_gui=lambda fn: fn())
    assert used == ["improve_dataset"] and "took it out" in answer
    result = json.loads(sent[1][-1]["content"][0]["content"])
    assert result["outcome"] == "taken out"
    assert intake.read_manifest(dataset, "newfilm") is None


def test_the_app_lends_the_chat_the_dataset_tools():
    names = [t.name for t in aa.app_tools(object(), call_on_gui=lambda fn: fn(),
                                          cancelled=lambda: False)]
    assert names[:4] == ["app_state", "read_report", "show_clips", "run_highlight"]
    assert "improve_dataset" in names and "train_action_model" in names
