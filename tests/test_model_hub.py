"""Offline tests for model_hub (no network, no Hugging Face account needed).

The tests build tiny ONNX models, so they need ``onnx`` and ``onnxruntime``;
without them (the light CI install) the module is skipped.
"""
import json
import os
import sys

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conftest import real_opencv  # noqa: E402

from model_hub import InputSpec, Manifest, ManifestError, build_package, check_package  # noqa: E402
from model_hub import hub  # noqa: E402
from model_hub.manifest import (  # noqa: E402
    COMPLIANCE_ITEMS, category_from_tags, category_tags,
)
from model_hub.package import draft_for_trained_detector  # noqa: E402

ALL_OK = {k: True for k in COMPLIANCE_ITEMS}
SIZE = 64
ANCHORS = (SIZE // 8) ** 2 + (SIZE // 16) ** 2 + (SIZE // 32) ** 2   # 84


def _constant_output_model(path, input_shape, output_values, opset=17):
    """A graph whose output is a fixed tensor (plus 0 * the input), so a test
    controls exactly what a "detector" says."""
    const = numpy_helper.from_array(output_values.astype(np.float32), "C")
    zero = numpy_helper.from_array(np.array([0.0], np.float32), "Z")
    axes = numpy_helper.from_array(np.array(list(range(len(input_shape))), np.int64), "axes")
    nodes = [
        helper.make_node("ReduceMean", ["images", "axes"], ["m"], keepdims=0),
        helper.make_node("Mul", ["m", "Z"], ["mz"]),
        helper.make_node("Add", ["C", "mz"], ["output"]),
    ]
    graph = helper.make_graph(
        nodes, "g",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, list(input_shape))],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, list(output_values.shape))],
        [const, zero, axes])
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]), str(path))


def make_yolox_like(path, classes=2):
    """[1,3,64,64] -> [1, 84, 5 + classes], one confident box of class 0."""
    out = np.zeros((1, ANCHORS, 5 + classes), np.float32)
    out[0, 0, :5] = [0.5, 0.5, np.log(2.0), np.log(2.0), 0.95]
    out[0, 0, 5] = 0.95
    _constant_output_model(path, [1, 3, SIZE, SIZE], out)


def make_transposed(path, classes=2):
    """The AGPL toolkit's layout: [1, 4 + classes, anchors]."""
    _constant_output_model(path, [1, 3, SIZE, SIZE], np.zeros((1, 4 + classes, ANCHORS), np.float32))


def make_classifier(path, classes=3, size=32):
    w = numpy_helper.from_array(np.random.rand(classes, 3).astype(np.float32), "W")
    b = numpy_helper.from_array(np.zeros(classes, np.float32), "B")
    nodes = [
        helper.make_node("GlobalAveragePool", ["input"], ["pooled"]),
        helper.make_node("Flatten", ["pooled"], ["flat"]),
        helper.make_node("Gemm", ["flat", "W", "B"], ["logits"], transB=1),
    ]
    graph = helper.make_graph(
        nodes, "cls",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 3, size, size])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", classes])],
        [w, b])
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), str(path))


def det_manifest(**over):
    data = dict(
        name="test-detector", display_name="Test detector", description="Finds two shapes.",
        task="object_detection", labels=["a", "b"], author="tester", license="apache-2.0",
        category="test/shapes", output_format="yolox",
        input=InputSpec(width=SIZE, height=SIZE), compliance=dict(ALL_OK),
        metrics={"heldout_found": 11, "heldout_expected": 11, "rounds": 10})
    data.update(over)
    return Manifest(**data)


# --------------------------------------------------------------------------- #
# packaging
# --------------------------------------------------------------------------- #
def test_a_yolox_detector_package_passes(tmp_path):
    make_yolox_like(tmp_path / "m.onnx")
    out, report = build_package(tmp_path / "m.onnx", det_manifest(), tmp_path / "pkg")
    assert report.ok, report.text()
    assert sorted(p.name for p in out.iterdir()) == ["README.md", "model.onnx", "videohighlighter.json"]
    card = (out / "README.md").read_text(encoding="utf-8")
    assert "- videohighlighter" in card and "No clips, screenshots" in card
    assert "- vh-cat-test" in card and "- vh-cat-test--shapes" in card
    assert "Found 11 of 11 on frames it never trained on" in card


def test_the_transposed_agpl_layout_is_refused(tmp_path):
    make_transposed(tmp_path / "m.onnx")
    _, report = build_package(tmp_path / "m.onnx", det_manifest(), tmp_path / "pkg")
    assert not report.ok
    assert any("AGPL" in e for e in report.errors)


def test_label_count_mismatch_fails(tmp_path):
    make_yolox_like(tmp_path / "m.onnx", classes=2)
    _, report = build_package(tmp_path / "m.onnx", det_manifest(labels=["a", "b", "c"]), tmp_path / "pkg")
    assert any("does not match the YOLOX format" in e for e in report.errors)


@pytest.mark.parametrize("licence", ["agpl-3.0", "cc-by-sa-4.0", "other", "gpl-3.0", ""])
def test_restrictive_or_unknown_licences_are_refused(licence):
    assert any("license must be one of" in p for p in det_manifest(license=licence).problems())


@pytest.mark.parametrize("category", ["", "Animals", "a/b/c/d", "../x", "a//b", "a b"])
def test_a_bad_category_is_refused(category):
    assert any("category must" in p for p in det_manifest(category=category).problems())


def test_yolov8_is_no_longer_a_detector_format():
    assert any("output_format" in p for p in det_manifest(output_format="yolov8").problems())


def test_a_yolox_detector_must_take_bgr_0_255():
    m = det_manifest(input=InputSpec(width=SIZE, height=SIZE, color="RGB", normalize="0-1"))
    assert any("BGR, 0-255" in p for p in m.problems())


def test_metrics_must_be_known_numbers():
    assert det_manifest(metrics={"heldout_found": 3}).problems(require_compliance=False) == []
    assert any("Unknown metrics" in p for p in det_manifest(metrics={"path": 1}).problems())
    assert any("non-negative number" in p for p in det_manifest(metrics={"rounds": "C:/x"}).problems())


def test_category_tags_round_trip():
    assert category_tags("animals/horses/jumping") == [
        "vh-cat-animals", "vh-cat-animals--horses", "vh-cat-animals--horses--jumping"]
    assert category_from_tags(["videohighlighter", *category_tags("animals/horses")]) == "animals/horses"
    assert category_from_tags(["videohighlighter"]) == ""


def test_extra_files_are_rejected(tmp_path):
    make_yolox_like(tmp_path / "m.onnx")
    out, _ = build_package(tmp_path / "m.onnx", det_manifest(), tmp_path / "pkg")
    (out / "clip.mp4").write_bytes(b"x" * 100)
    (out / "frame.jpg").write_bytes(b"x")
    report, _ = check_package(out)
    assert sum("Not allowed" in e for e in report.errors) == 2


def test_tampered_model_is_rejected(tmp_path):
    make_yolox_like(tmp_path / "m.onnx")
    out, _ = build_package(tmp_path / "m.onnx", det_manifest(), tmp_path / "pkg")
    make_yolox_like(out / "model.onnx", classes=2)
    with open(out / "model.onnx", "ab") as fh:
        fh.write(b"\0")
    report, _ = check_package(out)
    assert any("sha256" in e for e in report.errors)


def test_checklist_required(tmp_path):
    make_yolox_like(tmp_path / "m.onnx")
    _, report = build_package(tmp_path / "m.onnx", det_manifest(compliance={"terms_checked": True}),
                              tmp_path / "pkg")
    assert any("checklist" in e for e in report.errors)


def test_non_onnx_refused(tmp_path):
    (tmp_path / "m.pt").write_bytes(b"x")
    with pytest.raises(ValueError):
        build_package(tmp_path / "m.pt", det_manifest(), tmp_path / "pkg")
    assert not (tmp_path / "pkg").exists()


def test_unknown_manifest_field_rejected():
    data = det_manifest().to_dict()
    data["run_script"] = "rm -rf /"
    with pytest.raises(ManifestError):
        Manifest.from_dict(data)


def test_manifest_roundtrip(tmp_path):
    m = det_manifest()
    m.save(tmp_path)
    assert Manifest.load(tmp_path) == m


def test_a_classifier_packages_but_this_app_will_not_install_it(tmp_path):
    make_classifier(tmp_path / "c.onnx")
    m = det_manifest(name="menu-screen", task="image_classification", labels=["x", "y", "z"],
                     output_format="logits", input=InputSpec(width=32, height=32, color="RGB",
                                                             normalize="0-1"))
    out, report = build_package(tmp_path / "c.onnx", m, tmp_path / "pkg")
    assert report.ok, report.text()
    model, report = hub.install_folder(out, "u/menu", "rev", tmp_path / "models")
    assert model is None and any("only use object detection" in e for e in report.errors)


# --------------------------------------------------------------------------- #
# the draft the Train tab hands the wizard
# --------------------------------------------------------------------------- #
def test_a_trained_detector_needs_only_name_description_and_category(tmp_path):
    folder = tmp_path / "yolox_tiny_custom"
    folder.mkdir()
    make_yolox_like(folder / "yolox_tiny_custom.onnx")
    (folder / "labels.json").write_text(json.dumps(["a", "b"]), encoding="utf-8")
    (folder / "yolox_tiny_custom.meta.json").write_text(
        json.dumps({"class_names": ["a", "b"], "input_size": [SIZE, SIZE]}), encoding="utf-8")

    draft = draft_for_trained_detector(folder / "yolox_tiny_custom.onnx", author="me",
                                       metrics={"heldout_found": 4, "heldout_expected": 5})
    assert draft.labels == ["a", "b"]
    assert (draft.input.width, draft.input.height) == (SIZE, SIZE)
    draft.name, draft.display_name = "shapes", "Shapes"
    draft.description, draft.category = "Finds shapes.", "test/shapes"
    draft.compliance = dict(ALL_OK)
    _, report = build_package(folder / "yolox_tiny_custom.onnx", draft, tmp_path / "pkg")
    assert report.ok, report.text()


# --------------------------------------------------------------------------- #
# sharing and installing
# --------------------------------------------------------------------------- #
def test_the_kraken_is_released_only_for_a_clean_package(tmp_path):
    make_yolox_like(tmp_path / "m.onnx")
    out, _ = build_package(tmp_path / "m.onnx", det_manifest(), tmp_path / "pkg")
    said = []
    manifest = hub.release_the_kraken(out, progress=said.append)
    assert manifest.name == "test-detector"
    assert said[-1].startswith(hub.KRAKEN + " Release the kraken")

    (out / "notes.txt").write_text("x")
    with pytest.raises(RuntimeError, match="did not pass"):
        hub.release_the_kraken(out, progress=lambda _m: None)


def test_the_kraken_survives_a_console_without_emoji(tmp_path):
    make_yolox_like(tmp_path / "m.onnx")
    out, _ = build_package(tmp_path / "m.onnx", det_manifest(), tmp_path / "pkg")
    said = []

    def cp1250(line):
        line.encode("cp1250")
        said.append(line)

    hub.release_the_kraken(out, progress=cp1250)
    assert "Release the kraken" in said[-1]


def test_install_lays_the_model_out_for_the_object_model_picker(tmp_path):
    make_yolox_like(tmp_path / "m.onnx")
    out, _ = build_package(tmp_path / "m.onnx", det_manifest(), tmp_path / "pkg")
    model, report = hub.install_folder(out, "someone/test-detector", "abc123", tmp_path / "models")
    assert model is not None, report.text()
    assert json.loads((model.path / "labels.json").read_text()) == ["a", "b"]
    assert hub.verify_installed(model).ok

    listed = hub.installed_detectors(tmp_path / "models")
    assert listed == [{"path": str(model.model_path), "name": "Test detector",
                       "classes": ["a", "b"], "community": "someone/test-detector"}]

    with open(model.model_path, "ab") as fh:        # changed on disk
        fh.write(b"\0")
    assert hub.installed_detectors(tmp_path / "models") == []


def test_blocklist_parsing_is_strict_and_case_insensitive():
    data = {"blocked": [{"repo_id": "Someone/Bad-Model", "reason": "report"},
                        {"repo_id": "not a repo id"}, "other/model", 42]}
    assert hub.parse_blocklist(data) == {"someone/bad-model", "other/model"}
    assert hub.parse_blocklist({"unexpected": True}) == set()


def test_an_unreachable_blocklist_fails_open():
    assert hub.fetch_blocklist("http://127.0.0.1:9/nothing.json", timeout=0.5) == set()


def test_a_blocked_repository_is_not_installed(monkeypatch):
    monkeypatch.setattr(hub, "fetch_blocklist", lambda *a, **k: {"someone/bad-model"})
    model, report = hub.install("Someone/Bad-Model")
    assert model is None and "removed from the community models" in report.errors[0]


def test_catalog_categories_include_their_parents():
    e = lambda cat: hub.CatalogEntry("u/m", "u", "object_detection", cat, 0, 0, "", "", [])  # noqa: E731
    assert hub.categories_in([e("animals/horses"), e("games"), e("")]) == [
        "animals", "animals/horses", "games"]


def test_the_community_runner_uses_the_apps_detector(tmp_path, monkeypatch):
    cv2 = real_opencv()
    if cv2 is None:
        pytest.skip("OpenCV not installed")
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    from model_hub.runtime import CommunityModelRunner

    make_yolox_like(tmp_path / "m.onnx")
    out, _ = build_package(tmp_path / "m.onnx", det_manifest(), tmp_path / "pkg")
    model, _ = hub.install_folder(out, "u/m", "rev", tmp_path / "models")
    runner = CommunityModelRunner(model)
    hits = runner.detect(np.full((360, 640, 3), 90, np.uint8))
    assert [h.label for h in hits] == ["a"]
    x1, y1, x2, y2 = hits[0].box
    assert 0 <= x1 < x2 <= 640 and 0 <= y1 < y2 <= 360
    assert runner.signal_name == "community:test-detector"


def test_the_runner_refuses_a_tampered_install(tmp_path):
    from model_hub.runtime import CommunityModelRunner, ModelCheckFailed
    make_yolox_like(tmp_path / "m.onnx")
    out, _ = build_package(tmp_path / "m.onnx", det_manifest(), tmp_path / "pkg")
    model, _ = hub.install_folder(out, "u/m", "rev", tmp_path / "models")
    with open(model.model_path, "ab") as fh:
        fh.write(b"\0")
    with pytest.raises(ModelCheckFailed):
        CommunityModelRunner(model)
