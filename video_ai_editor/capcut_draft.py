"""Edit-timeline export as a CapCut / Jianying project ("draft") folder.

CapCut and Jianying keep each project as a folder of JSON under their
drafts root. Neither documents the format and neither imports EDL or
FCPXML, so this writes the folder itself. The app finds a new folder in
its drafts root on its own; nothing else needs registering.

Only the fields the app cannot work out are written: the media file, and
each clip's source and timeline range. When the project is opened the app
fills in the rest (effects slots, colour and audio settings) and saves the
folder in its own full format.

Times are microseconds. They are counted from real frames, so a clip
starts and ends on the same frame as in the EDL and FCPXML exports.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
import uuid

from video_ai_editor.timeline_export import (
    ExportError,
    Sequence,
    WrittenExport,
    _source_clip,
    _stem,
)

CAPCUT = "capcut"
JIANYING = "jianying"

# Platform block per app. The schema versions are deliberately old: an app
# upgrades an older project when it opens it, and may refuse a newer one.
_APPS = {
    CAPCUT: {"label": "CapCut", "folder": "CapCut", "app_id": 359289,
             "app_source": "cc"},
    JIANYING: {"label": "Jianying", "folder": "JianyingPro", "app_id": 3704,
               "app_source": "lv"},
}
_DRAFT_VERSION = 360000
_DRAFT_NEW_VERSION = "110.0.0"
_DRAFT_APP_VERSION = "5.9.0"

# Every material list the app expects to find, even when empty.
_MATERIAL_KINDS = (
    "audio_balances", "audio_effects", "audio_fades", "audio_track_indexes",
    "audios", "beats", "canvases", "chromas", "color_curves", "drafts",
    "effects", "flowers", "green_screens", "handwrites", "hsl", "images",
    "log_color_wheels", "loudnesses", "manual_deformations", "material_animations",
    "material_colors", "placeholders", "plugin_effects", "primary_color_wheels",
    "realtime_denoises", "shapes", "smart_crops", "sound_channel_mappings",
    "speeds", "stickers", "tail_leaders", "text_templates", "texts",
    "time_marks", "transitions", "video_effects", "video_trackings", "videos",
    "vocal_beautifys", "vocal_separations",
)
_KEYFRAME_KINDS = (
    "adjusts", "audios", "effects", "filters", "handwrites", "stickers",
    "texts", "videos",
)


def _uid() -> str:
    return str(uuid.uuid4()).upper()


def frames_to_us(frames: int, fps_num: int, fps_den: int) -> int:
    """Microseconds at the start of frame ``frames`` on ``fps_num/fps_den``."""
    return int(round(frames * fps_den * 1_000_000 / fps_num))


def drafts_root(app: str, home: str | None = None,
                local_app_data: str | None = None) -> str:
    """Where ``app`` keeps its projects on this platform. It may not exist."""
    folder = _APPS[app]["folder"]
    tail = os.path.join("User Data", "Projects", "com.lveditor.draft")
    if sys.platform == "win32":
        base = local_app_data or os.environ.get("LOCALAPPDATA") or ""
        return os.path.join(base, folder, tail) if base else ""
    if sys.platform == "darwin":
        base = home or os.path.expanduser("~")
        return os.path.join(base, "Movies", folder, tail)
    return ""


def installed_drafts_roots(home: str | None = None,
                           local_app_data: str | None = None) -> list[tuple[str, str]]:
    """``(app, root)`` for each app whose drafts root exists, CapCut first."""
    found = []
    for app in (CAPCUT, JIANYING):
        root = drafts_root(app, home=home, local_app_data=local_app_data)
        if root and os.path.isdir(root):
            found.append((app, root))
    return found


def app_label(app: str) -> str:
    return _APPS[app]["label"]


def _platform(app: str) -> dict:
    return {
        "os": "mac" if sys.platform == "darwin" else "windows",
        "os_version": "",
        "app_id": _APPS[app]["app_id"],
        "app_version": _DRAFT_APP_VERSION,
        "app_source": _APPS[app]["app_source"],
        "device_id": "",
        "hard_disk_id": "",
        "mac_address": "",
    }


def _video_material(sequence: Sequence) -> dict:
    source = sequence.source
    path = os.path.abspath(source.path).replace("\\", "/")
    return {
        "id": _uid(),
        "type": "video",
        "path": path,
        "material_name": os.path.basename(path),
        "duration": int(round(source.duration * 1_000_000)),
        "width": source.width,
        "height": source.height,
        "has_audio": source.has_audio,
        "category_name": "local",
        "category_id": "",
        "check_flag": 63487,
        "crop": {"upper_left_x": 0.0, "upper_left_y": 0.0,
                 "upper_right_x": 1.0, "upper_right_y": 0.0,
                 "lower_left_x": 0.0, "lower_left_y": 1.0,
                 "lower_right_x": 1.0, "lower_right_y": 1.0},
        "crop_ratio": "free",
        "crop_scale": 1.0,
        "extra_type_option": 0,
        "local_material_id": "",
        "media_path": "",
        "source": 0,
        "source_platform": 0,
    }


def _segment(material_id: str, refs: list[str],
             source_range: tuple[int, int], target_range: tuple[int, int]) -> dict:
    return {
        "id": _uid(),
        "material_id": material_id,
        "source_timerange": {"start": source_range[0], "duration": source_range[1]},
        "target_timerange": {"start": target_range[0], "duration": target_range[1]},
        "speed": 1.0,
        "volume": 1.0,
        "last_nonzero_volume": 1.0,
        "visible": True,
        "reverse": False,
        "render_index": 0,
        "track_render_index": 0,
        "track_attribute": 0,
        "clip": {"alpha": 1.0, "rotation": 0.0,
                 "flip": {"horizontal": False, "vertical": False},
                 "scale": {"x": 1.0, "y": 1.0},
                 "transform": {"x": 0.0, "y": 0.0}},
        "uniform_scale": {"on": True, "value": 1.0},
        "hdr_settings": {"intensity": 1.0, "mode": 1, "nits": 1000},
        "extra_material_refs": refs,
        "common_keyframes": [],
        "keyframe_refs": [],
        "enable_adjust": True,
        "enable_color_curves": True,
        "enable_color_wheels": True,
        "enable_lut": True,
        "enable_video_mask": True,
        "intensifies_audio": False,
        "is_placeholder": False,
        "is_tone_modify": False,
        "cartoon": False,
        "group_id": "",
        "template_id": "",
        "template_scene": "default",
    }


def draft_content(sequence: Sequence, app: str = CAPCUT,
                  timeline_id: str | None = None) -> tuple[dict, int]:
    """The project's ``draft_content.json``, and how many clips had no frames.

    Clips play back to back on one video track; the video's own audio comes
    with them. Each clip has its own copy of the video material, as the app
    does, so a crop or effect on one clip leaves the others alone.
    """
    if app not in _APPS:
        raise ExportError(f"unknown app {app!r}")
    timebase = sequence.timebase()
    num, den = timebase.fps_num, timebase.fps_den
    materials = {kind: [] for kind in _MATERIAL_KINDS}
    segments = []
    skipped = 0
    cursor = 0
    for start, end in sequence.clips:
        frames = _source_clip(start, end, timebase)
        if frames is None:
            skipped += 1
            continue
        src_in, src_out = frames
        length = src_out - src_in
        source_start = frames_to_us(src_in, num, den)
        source_length = frames_to_us(src_out, num, den) - source_start
        target_start = frames_to_us(cursor, num, den)
        target_length = frames_to_us(cursor + length, num, den) - target_start
        cursor += length

        video = _video_material(sequence)
        speed = {"id": _uid(), "type": "speed", "mode": 0, "speed": 1.0,
                 "curve_speed": None}
        canvas = {"id": _uid(), "type": "canvas_color", "color": "",
                  "blur": 0.0, "image": "", "album_image": "", "image_id": "",
                  "image_name": ""}
        materials["videos"].append(video)
        materials["speeds"].append(speed)
        materials["canvases"].append(canvas)
        segments.append(_segment(
            video["id"], [speed["id"], canvas["id"]],
            (source_start, source_length), (target_start, target_length)))

    if not segments:
        raise ExportError("no clip on the edit timeline is a whole frame long")

    now = int(time.time())
    content = {
        "id": timeline_id or _uid(),
        "version": _DRAFT_VERSION,
        "new_version": _DRAFT_NEW_VERSION,
        "name": "",
        "duration": frames_to_us(cursor, num, den),
        "create_time": now,
        "update_time": now,
        "fps": num / den,
        "is_drop_frame_timecode": False,
        "color_space": -1,
        "config": {
            "video_mute": False,
            "record_audio_last_index": 1,
            "extract_audio_last_index": 1,
            "original_sound_last_index": 1,
            "subtitle_recognition_id": "",
            "lyrics_recognition_id": "",
            "subtitle_sync": True,
            "lyrics_sync": True,
            "sticker_max_index": 1,
            "adjust_max_index": 1,
            "material_save_mode": 0,
            "export_range": None,
            "maintrack_adsorb": True,
            "combination_max_index": 1,
            "attachment_info": [],
        },
        "canvas_config": {"ratio": "original", "width": sequence.source.width,
                          "height": sequence.source.height, "background": None},
        "tracks": [{"id": _uid(), "type": "video", "attribute": 0, "flag": 0,
                    "is_default_name": True, "name": "", "segments": segments}],
        "group_container": None,
        "materials": materials,
        "keyframes": {kind: [] for kind in _KEYFRAME_KINDS},
        "keyframe_graph_list": [],
        "platform": _platform(app),
        "last_modified_platform": _platform(app),
        "mutable_config": None,
        "cover": None,
        "retouch_cover": None,
        "extra_info": None,
        "relationships": [],
        "free_render_index_mode_on": False,
        "render_index_track_mode_on": True,
        "static_cover_image_path": "",
        "source": "default",
        "time_marks": None,
        "path": "",
        "draft_type": "video",
    }
    return content, skipped


def _unique_name(root: str, name: str) -> str:
    candidate, number = name, 2
    while os.path.exists(os.path.join(root, candidate)):
        candidate = f"{name} ({number})"
        number += 1
    return candidate


def _dump(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))


def write_draft(sequence: Sequence, root: str, app: str = CAPCUT,
                name: str | None = None) -> WrittenExport:
    """Write a project folder under ``root`` and return its path.

    The name defaults to ``{stem} highlights``; a name already taken gets
    `` (2)``, `` (3)``... The folder is built under a dot-name first and
    renamed when complete, so the app never lists a half-written project.
    """
    if not root:
        raise ExportError("no folder to write the project into")
    os.makedirs(root, exist_ok=True)
    name = _unique_name(root, name or f"{_stem(sequence.source.path)} highlights")
    folder = os.path.join(root, name)
    partial = os.path.join(root, f".{name}.part")

    timeline_id = _uid()
    content, skipped = draft_content(sequence, app, timeline_id)
    now_us = int(time.time() * 1_000_000)
    meta = {
        "draft_fold_path": folder.replace("\\", "/"),
        "draft_root_path": root,
        "draft_id": _uid(),
        "draft_name": name,
        "draft_cover": "",
        "draft_is_invisible": False,
        "draft_materials": [{"type": kind, "value": []} for kind in (0, 1, 2, 3, 6)],
        "draft_materials_copied_info": [],
        "draft_segment_extra_info": [],
        "draft_new_version": "",
        "draft_type": "",
        "draft_timeline_materials_size_": 0,
        "tm_draft_create": now_us,
        "tm_draft_modified": now_us,
        "tm_draft_removed": 0,
        "tm_duration": content["duration"],
    }
    project = {
        "config": {"color_space": -1, "hdr_vivid": False,
                   "mixed_track_mode_on": False,
                   "render_index_track_mode_on": False,
                   "use_float_render": False},
        "create_time": now_us,
        "id": _uid(),
        "main_timeline_id": timeline_id,
        "timelines": [{"create_time": now_us, "id": timeline_id,
                       "is_marked_delete": False, "name": name,
                       "update_time": now_us}],
        "update_time": now_us,
        "version": 0,
    }
    layout = {"dockItems": [{"dockIndex": 0, "ratio": 1,
                             "timelineIds": [timeline_id],
                             "timelineNames": [name]}],
              "layoutOrientation": 1}

    shutil.rmtree(partial, ignore_errors=True)
    try:
        _dump(os.path.join(partial, "draft_content.json"), content)
        # Newer versions read the timeline's own copy; older ones the top one.
        _dump(os.path.join(partial, "Timelines", timeline_id, "draft_content.json"),
              content)
        _dump(os.path.join(partial, "Timelines", "project.json"), project)
        _dump(os.path.join(partial, "draft_meta_info.json"), meta)
        _dump(os.path.join(partial, "timeline_layout.json"), layout)
        os.replace(partial, folder)
    except Exception:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return WrittenExport(path=folder, skipped=skipped)
