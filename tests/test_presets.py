"""Named settings live beside config.yaml and do not replace it."""
from pathlib import Path

from modules.system.presets import (
    delete_preset,
    list_presets,
    load_preset,
    safe_preset_name,
    save_preset,
)


def test_safe_names_cannot_leave_the_presets_folder():
    assert safe_preset_name("  drawing timelapse ") == "drawing timelapse"
    assert safe_preset_name("../config") == "config"
    assert safe_preset_name("a/b") == "a b"
    assert safe_preset_name("...") == ""
    assert safe_preset_name("") == ""


def test_save_list_load_delete_round_trip(tmp_path: Path):
    config = tmp_path / "config.yaml"
    config.write_text("highlights:\n  output: highlight.mp4\n", encoding="utf-8")
    data = {"highlights": {"output": "reel.mp4", "clip_time": 8}, "scoring": {"scene_points": 2}}

    path = save_preset("drawing timelapse", data, str(config))
    assert Path(path).parent == tmp_path / "presets"
    assert list_presets(str(config)) == ["drawing timelapse"]
    assert load_preset("drawing timelapse", str(config))["highlights"]["output"] == "reel.mp4"
    # The live file is not rewritten by the store itself.
    assert "reel.mp4" not in config.read_text(encoding="utf-8")

    assert delete_preset("drawing timelapse", str(config)) is True
    assert list_presets(str(config)) == []
    assert config.exists()
    assert delete_preset("drawing timelapse", str(config)) is False


def test_output_field_and_tooltips_are_wired():
    root = Path(__file__).resolve().parents[1]
    main = (root / "main.py").read_text(encoding="utf-8")
    pipeline = (root / "pipeline.py").read_text(encoding="utf-8")
    assert "output_base" in pipeline
    assert "highlight_output_path" in pipeline
    for snippet in (
        "self.output_input.setToolTip",
        "self.spin_scene_points.setToolTip",
        "self.spin_motion_event_points.setToolTip",
        "self.spin_motion_peak.setToolTip",
        "self.spin_audio_peak.setToolTip",
        "self.spin_loudness_burst.setToolTip",
        "self.spin_keyword_points.setToolTip",
        "self.spin_transcript_points.setToolTip",
        "self.spin_object.setToolTip",
        "self.spin_action.setToolTip",
        "self.spin_face_expression.setToolTip",
        "self.spin_beginning_seconds.setToolTip",
        "self.spin_beginning_points.setToolTip",
        "self.spin_ending_seconds.setToolTip",
        "self.spin_ending_points.setToolTip",
        "self.spin_max_duration.setToolTip",
        "self.spin_exact_duration.setToolTip",
        "self.spin_clip_time.setToolTip",
        "self.slider_coverage.setToolTip",
        "self.spin_auto_min_clip.setToolTip",
        "self.spin_auto_max_clip.setToolTip",
        "self.spin_auto_merge_gap.setToolTip",
        "self.export_clips_chk.setToolTip",
        "self.frame_skip_spin.setToolTip",
        "self.obj_frame_skip_spin.setToolTip",
        "self.render_mode_combo.setToolTip",
        "def save_named_preset",
        "def load_named_preset",
        "def delete_named_preset",
    ):
        assert snippet in main, snippet
