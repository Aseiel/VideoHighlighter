"""CapCut / Jianying project export: the draft folder the app opens.

No media file, no Qt, no CapCut. The shape is the one CapCut 9.6 opened and
played on 2026-10-28: one video track, clips back to back, times in
microseconds on real frame boundaries.
"""

from __future__ import annotations

import json
import os

import pytest

from video_ai_editor import capcut_draft
from video_ai_editor.capcut_draft import (
    CAPCUT,
    JIANYING,
    draft_content,
    drafts_root,
    frames_to_us,
    installed_drafts_roots,
    write_draft,
)
from video_ai_editor.timeline_export import (
    CAPCUT_FORMAT,
    RECORD_START_HOUR,
    RECORD_START_ZERO,
    ExportError,
    MediaSource,
    Sequence,
    TimelineExporter,
    has_record_start,
    record_start_for_format,
)


def _sequence(clips, num=30000, den=1001, path="clip.mp4") -> Sequence:
    source = MediaSource(
        path=path, duration=60.0, fps_num=num, fps_den=den,
        width=1280, height=720, has_audio=True, audio_rate=48000,
        audio_channels=2)
    return Sequence(title="clip", source=source, clips=tuple(clips))


def _segments(content):
    (track,) = content["tracks"]
    assert track["type"] == "video"
    return track["segments"]


# ---------------------------------------------------------------------------
# Times
# ---------------------------------------------------------------------------

def test_cuts_land_on_real_frames_in_microseconds():
    """5 s at 29.97 is frame 150, which starts at 5.005 s."""
    content, skipped = draft_content(_sequence([(5.0, 10.0), (20.0, 23.5)]))

    first, second = _segments(content)
    assert skipped == 0
    assert first["source_timerange"] == {"start": 5_005_000, "duration": 5_005_000}
    assert second["source_timerange"]["start"] == frames_to_us(599, 30000, 1001)
    assert frames_to_us(150, 30000, 1001) == 5_005_000


def test_clips_play_back_to_back_and_fill_the_project():
    content, _ = draft_content(_sequence([(5.0, 10.0), (20.0, 23.5), (40.0, 48.0)]))

    cursor = 0
    for segment in _segments(content):
        target = segment["target_timerange"]
        assert target["start"] == cursor
        # Same frame count; the two microsecond roundings may differ by one.
        assert abs(target["duration"] - segment["source_timerange"]["duration"]) <= 1
        cursor += target["duration"]
    assert content["duration"] == cursor
    assert content["fps"] == pytest.approx(30000 / 1001)


def test_a_clip_shorter_than_a_frame_is_skipped_and_counted():
    content, skipped = draft_content(_sequence([(1.0, 1.001), (2.0, 3.0)], 30, 1))

    assert skipped == 1
    assert len(_segments(content)) == 1


def test_nothing_to_write_is_an_error():
    with pytest.raises(ExportError, match="whole frame"):
        draft_content(_sequence([(1.0, 1.001)], 30, 1))


# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------

def test_each_clip_has_its_own_video_material_and_its_refs_resolve():
    content, _ = draft_content(_sequence([(0.0, 1.0), (2.0, 3.0)], path="morning.mp4"))
    materials = content["materials"]
    by_id = {item["id"]: kind for kind, items in materials.items() for item in items}

    segments = _segments(content)
    assert len({segment["material_id"] for segment in segments}) == 2
    for segment in segments:
        assert by_id[segment["material_id"]] == "videos"
        assert sorted(by_id[ref] for ref in segment["extra_material_refs"]) == [
            "canvases", "speeds"]
    video = materials["videos"][0]
    assert video["path"] == os.path.abspath("morning.mp4").replace("\\", "/")
    assert video["material_name"] == "morning.mp4"
    assert (video["width"], video["height"], video["has_audio"]) == (1280, 720, True)


def test_the_project_carries_no_machine_identity():
    content, _ = draft_content(_sequence([(0.0, 1.0)]))

    for key in ("platform", "last_modified_platform"):
        assert content[key]["device_id"] == ""
        assert content[key]["mac_address"] == ""
        assert content[key]["hard_disk_id"] == ""


def test_jianying_is_named_as_jianying():
    capcut, _ = draft_content(_sequence([(0.0, 1.0)]), CAPCUT)
    jianying, _ = draft_content(_sequence([(0.0, 1.0)]), JIANYING)

    assert capcut["platform"]["app_source"] == "cc"
    assert jianying["platform"]["app_source"] == "lv"
    with pytest.raises(ExportError, match="unknown app"):
        draft_content(_sequence([(0.0, 1.0)]), "premiere")


# ---------------------------------------------------------------------------
# The folder
# ---------------------------------------------------------------------------

def _load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def test_the_folder_has_what_capcut_reads(tmp_path):
    result = write_draft(_sequence([(5.0, 10.0)], path=str(tmp_path / "morning.mp4")),
                         str(tmp_path))

    folder = tmp_path / "morning highlights"
    assert result.path == str(folder)
    assert result.skipped == 0
    content = _load(folder / "draft_content.json")
    project = _load(folder / "Timelines" / "project.json")
    meta = _load(folder / "draft_meta_info.json")
    layout = _load(folder / "timeline_layout.json")

    assert project["main_timeline_id"] == content["id"]
    assert _load(folder / "Timelines" / content["id"] / "draft_content.json") == content
    assert meta["draft_name"] == "morning highlights"
    assert meta["draft_fold_path"] == str(folder).replace("\\", "/")
    assert meta["tm_duration"] == content["duration"]
    assert layout["dockItems"][0]["timelineIds"] == [content["id"]]
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".")] == []


def test_a_taken_name_gets_a_number(tmp_path):
    sequence = _sequence([(0.0, 1.0)], path=str(tmp_path / "morning.mp4"))

    first = write_draft(sequence, str(tmp_path))
    second = write_draft(sequence, str(tmp_path))

    assert os.path.basename(first.path) == "morning highlights"
    assert os.path.basename(second.path) == "morning highlights (2)"


def test_a_failed_write_leaves_no_folder(tmp_path, monkeypatch):
    calls = []
    real_dump = capcut_draft._dump

    def failing(path, payload):
        calls.append(path)
        if len(calls) == 3:
            raise OSError("disk full")
        real_dump(path, payload)

    monkeypatch.setattr(capcut_draft, "_dump", failing)
    with pytest.raises(OSError, match="disk full"):
        write_draft(_sequence([(0.0, 1.0)], path=str(tmp_path / "morning.mp4")),
                    str(tmp_path))
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# Where the apps keep projects
# ---------------------------------------------------------------------------

def test_only_installed_apps_are_offered_capcut_first(tmp_path, monkeypatch):
    monkeypatch.setattr(capcut_draft.sys, "platform", "win32")
    local = str(tmp_path)

    assert installed_drafts_roots(local_app_data=local) == []

    jianying = drafts_root(JIANYING, local_app_data=local)
    os.makedirs(jianying)
    assert installed_drafts_roots(local_app_data=local) == [(JIANYING, jianying)]

    capcut = drafts_root(CAPCUT, local_app_data=local)
    os.makedirs(capcut)
    assert installed_drafts_roots(local_app_data=local) == [
        (CAPCUT, capcut), (JIANYING, jianying)]
    assert capcut.endswith(os.path.join(
        "CapCut", "User Data", "Projects", "com.lveditor.draft"))


def test_on_a_mac_projects_live_under_movies(tmp_path, monkeypatch):
    monkeypatch.setattr(capcut_draft.sys, "platform", "darwin")

    assert drafts_root(CAPCUT, home=str(tmp_path)) == os.path.join(
        str(tmp_path), "Movies", "CapCut", "User Data", "Projects",
        "com.lveditor.draft")


# ---------------------------------------------------------------------------
# The exporter
# ---------------------------------------------------------------------------

def test_a_capcut_project_has_no_sequence_start():
    assert not has_record_start(CAPCUT_FORMAT)
    assert has_record_start("FCPXML")
    assert record_start_for_format(CAPCUT_FORMAT, RECORD_START_HOUR) == RECORD_START_ZERO


def test_without_capcut_installed_the_project_goes_beside_the_video(tmp_path, monkeypatch):
    source = tmp_path / "morning.mp4"
    source.write_bytes(b"")
    monkeypatch.setattr(capcut_draft, "installed_drafts_roots", lambda: [])

    result = TimelineExporter.to_capcut(
        [(0.0, 1.0)], str(source),
        source=_sequence([], path=str(source)).source)

    assert result.path == str(tmp_path / "morning highlights")


def test_export_auto_sends_capcut_to_its_writer(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(TimelineExporter, "to_capcut",
                        lambda *a, **k: calls.append("capcut"))

    TimelineExporter.export_auto([(0.0, 1.0)], str(tmp_path / "a.mp4"), format="capcut")
    TimelineExporter.export_auto([(0.0, 1.0)], str(tmp_path / "a.mp4"), format="jianying")

    assert calls == ["capcut", "capcut"]


def test_the_dialog_writes_a_capcut_project_without_a_save_dialog():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "signal_timeline_viewer.py"), encoding="utf-8") as fh:
        source = fh.read()
    export = source[source.index("def on_export_clicked"):
                    source.index("def _export_capcut_project")]

    assert export.index("CAPCUT_FORMAT") < export.index("getSaveFileName")
    assert "has_record_start" in export
