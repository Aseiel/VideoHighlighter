"""
llm_timeline_bridge.py — Bridge between LLM chat and Timeline Viewer.

Parses structured commands from LLM responses and executes them
on the SignalTimelineWindow's edit timeline, filters, and playback.

Command format (embedded in LLM response text):
    [CMD:add_clip start=10.5 end=15.0]
    [CMD:remove_clip index=2]
    [CMD:remove_clip index=all]
    [CMD:seek time=45.0]
    [CMD:play start=10 end=20]
    [CMD:filter_action name=drinking show=true]
    [CMD:filter_object name=person show=false]
    [CMD:show_all_filters]
    [CMD:confidence min=0.5 max=1.0]
    [CMD:zoom level=80]
    [CMD:list_clips]
    [CMD:clear_clips]
    [CMD:save]
    [CMD:export format=edl]

Usage:
    bridge = TimelineBridge(timeline_window)
    response = "I'll add that clip for you. [CMD:add_clip start=10.5 end=15.0]"
    clean_text, results = bridge.process_response(response)
"""

from __future__ import annotations

import re
from typing import Optional

# Command regex: [CMD:command_name key=value key=value ...]
# Strict: command name must be a known command word, not random text
CMD_PATTERN = re.compile(r'\[CMD:(\w+)((?:\s+\w+=\S+)*)\s*\]')
PARAM_PATTERN = re.compile(r'(\w+)=(\S+)')

KNOWN_COMMANDS = {
    'add_clip', 'remove_clip', 'clear_clips', 'seek', 'play', 'pause',
    'resume', 'filter_action', 'filter_object', 'show_all_filters',
    'confidence', 'zoom', 'list_clips', 'save', 'export',
}


def parse_commands(text: str) -> list[tuple[str, dict[str, str]]]:
    """
    Extract all [CMD:...] blocks from text.
    Returns list of (command_name, {param: value}) tuples.
    Only returns known commands to avoid false positives.
    """
    commands = []
    seen = set()  # Deduplicate identical commands
    for match in CMD_PATTERN.finditer(text):
        cmd_name = match.group(1)
        if cmd_name not in KNOWN_COMMANDS:
            continue
        params_str = match.group(2).strip()
        params = {}
        for pmatch in PARAM_PATTERN.finditer(params_str):
            params[pmatch.group(1)] = pmatch.group(2)

        # Deduplicate: skip if we already have the exact same command + params
        key = f"{cmd_name}:{sorted(params.items())}"
        if key in seen:
            continue
        seen.add(key)

        commands.append((cmd_name, params))
    return commands


def strip_commands(text: str) -> str:
    """Remove all [CMD:...] blocks from text, leaving clean response."""
    return CMD_PATTERN.sub('', text).strip()


class TimelineBridge:
    """
    Connects the LLM chat to the timeline viewer.
    Parses commands from LLM output and executes them.
    """

    def __init__(self, timeline_window=None):
        self._window = timeline_window  # SignalTimelineWindow instance
        self._command_log: list[str] = []

    def set_timeline_window(self, window):
        """Connect to a timeline window."""
        self._window = window

    @property
    def is_connected(self) -> bool:
        return self._window is not None

    def get_timeline_state(self) -> str:
        """
        Build a text summary of the current timeline state.
        This gets injected into the LLM prompt so it knows what's on the timeline.
        """
        if not self._window:
            return "Timeline: NOT CONNECTED"

        parts = ["## Current Timeline State"]

        # Edit clips
        w = self._window
        if hasattr(w, 'edit_scene') and w.edit_scene:
            clips = w.edit_scene.clips
            total_dur = w.edit_scene.get_total_duration()
            parts.append(f"Edit timeline: {len(clips)} clips, total {total_dur:.1f}s")
            for i, (s, e) in enumerate(clips):
                parts.append(f"  Clip {i+1}: {s:.1f}s - {e:.1f}s ({e-s:.1f}s)")
        else:
            parts.append("Edit timeline: empty")

        # Current time
        if hasattr(w, 'current_time'):
            parts.append(f"Playhead: {w.current_time:.1f}s")

        # Video info
        if hasattr(w, 'video_duration'):
            parts.append(f"Video duration: {w.video_duration:.1f}s")

        # Active filters
        if hasattr(w, 'signal_scene') and w.signal_scene:
            scene = w.signal_scene
            hidden_actions = [a for a, v in scene.visible_actions.items() if not v]
            hidden_objects = [o for o, v in scene.visible_objects.items() if not v]
            if hidden_actions:
                parts.append(f"Hidden actions: {', '.join(hidden_actions)}")
            if hidden_objects:
                parts.append(f"Hidden objects: {', '.join(hidden_objects)}")
            if scene.min_confidence > 0 or scene.max_confidence < 1:
                parts.append(f"Confidence filter: {scene.min_confidence:.2f} - {scene.max_confidence:.2f}")

        return "\n".join(parts)

    def get_available_commands_text(self) -> str:
        """
        Return the command reference for the LLM system prompt.
        """
        return """
## TIMELINE COMMANDS
You can control the timeline by including commands in your response.
Commands use this format: [CMD:command_name param1=value param2=value]

Available commands:
  [CMD:add_clip start=SECONDS end=SECONDS]     — Add a clip to the edit timeline
  [CMD:remove_clip index=NUMBER]                — Remove clip by number (1-based), or index=all to clear
  [CMD:clear_clips]                             — Remove all clips from edit timeline
  [CMD:seek time=SECONDS]                       — Move playhead to timestamp (does NOT play)
  [CMD:play]                                    — Play video from current position
  [CMD:play start=SECONDS end=SECONDS]          — Play a specific clip
  [CMD:pause]                                   — Pause video playback
  [CMD:filter_action name=ACTION_NAME show=true/false]  — Show/hide an action type
  [CMD:filter_object name=OBJECT_NAME show=true/false]  — Show/hide an object type
  [CMD:show_all_filters]                        — Reset all filters to show everything
  [CMD:confidence min=0.0 max=1.0]              — Set confidence threshold
  [CMD:zoom level=NUMBER]                       — Set zoom (10-200, default ~50)
  [CMD:list_clips]                              — List current edit timeline clips
  [CMD:save]                                    — Save edit timeline to cache
  [CMD:export format=edl]                       — Export timeline (edl or xml)

RULES for commands:
- Place commands AFTER your text explanation
- Use real timestamps from the analysis data
- For action/object names, use Title Case exactly as shown in the data
- You can include multiple commands in one response
- KEEP RESPONSES SHORT. Just say what you're doing + the command. Do NOT list all clips or echo back the timeline state.
- To play video, use [CMD:play] (from current pos) or [CMD:play start=X end=Y] (specific clip)
- [CMD:seek] only moves the playhead, it does NOT start playback
- BAD example (too verbose): "Here are your clips: Clip 1: 0-5s, Clip 2: 10-15s... I'll remove clip 2 [CMD:remove_clip index=2]"
- GOOD example (concise): "Removing clip 2. [CMD:remove_clip index=2]"

Example:
  "I see the action 'Drinking' at 0:10. I'll add it as a clip and play it.
  [CMD:add_clip start=8.0 end=13.0]
  [CMD:play start=8.0 end=13.0]"

  "Sure, playing the video now. [CMD:play]"
"""

    def process_response(self, response_text: str) -> tuple[str, list[str]]:
        """
        Process an LLM response: extract commands, execute them, return clean text + results.

        Returns:
            (clean_text, list_of_result_messages)
        """
        commands = parse_commands(response_text)
        clean_text = strip_commands(response_text)
        results = []

        for cmd_name, params in commands:
            result = self._execute_command(cmd_name, params)
            results.append(result)
            self._command_log.append(f"{cmd_name}: {result}")

        return clean_text, results

    def _execute_command(self, cmd: str, params: dict) -> str:
        """Execute a single command. Returns result message."""
        if not self._window:
            return f"⚠️ Timeline not connected — cannot execute {cmd}"

        try:
            handler = getattr(self, f'_cmd_{cmd}', None)
            if handler:
                return handler(params)
            else:
                return f"⚠️ Unknown command: {cmd}"
        except Exception as e:
            return f"❌ Error executing {cmd}: {e}"

    # ----------------------------------------------------------------
    # Command handlers
    # ----------------------------------------------------------------

    def _cmd_add_clip(self, p: dict) -> str:
        start = float(p.get('start', 0))
        end = float(p.get('end', start + 5))
        if end <= start:
            end = start + 3
        self._window.edit_scene.add_clip(start, end)
        self._window.update_edit_duration()
        return f"✅ Added clip: {start:.1f}s - {end:.1f}s"

    def _cmd_remove_clip(self, p: dict) -> str:
        idx_str = p.get('index', '').strip()
        if not idx_str:
            return "⚠️ Missing clip index"
        if idx_str.lower() == 'all':
            return self._cmd_clear_clips(p)

        try:
            idx = int(idx_str) - 1  # Convert 1-based to 0-based
        except ValueError:
            return f"⚠️ Invalid clip index: '{idx_str}'"

        clips = self._window.edit_scene.clips
        if 0 <= idx < len(clips):
            removed = clips[idx]
            clips.pop(idx)
            self._window.edit_scene.build_timeline()
            self._window.update_edit_duration()
            return f"✅ Removed clip {idx+1} ({removed[0]:.1f}s - {removed[1]:.1f}s)"
        else:
            return f"⚠️ Invalid clip index {idx+1} (have {len(clips)} clips)"

    def _cmd_clear_clips(self, p: dict) -> str:
        count = len(self._window.edit_scene.clips)
        self._window.edit_scene.clips.clear()
        self._window.edit_scene.build_timeline()
        self._window.update_edit_duration()
        return f"✅ Cleared all {count} clips"

    def _cmd_seek(self, p: dict) -> str:
        t = float(p.get('time', 0))
        t = max(0, min(t, self._window.video_duration))
        self._window.on_time_clicked(t)
        return f"✅ Seeked to {t:.1f}s"

    def _cmd_play(self, p: dict) -> str:
        start = p.get('start')
        end = p.get('end')

        # If no params, play from current position
        if start is None and end is None:
            if hasattr(self._window, 'video_player'):
                current = getattr(self._window, 'current_time', 0)
                self._window.video_player.setPosition(int(current * 1000))
                self._window.video_player.play()
                if hasattr(self._window, 'play_btn'):
                    self._window.play_btn.setText("⏸ Pause")
                return f"✅ Playing from {current:.1f}s"
            return "⚠️ No video player available"

        start = float(start)
        end = float(end) if end else start + 5
        self._window.play_video_clip(start, end)
        return f"✅ Playing {start:.1f}s - {end:.1f}s"

    def _cmd_pause(self, p: dict) -> str:
        if hasattr(self._window, 'video_player'):
            self._window.video_player.pause()
            if hasattr(self._window, 'play_btn'):
                self._window.play_btn.setText("▶ Play")
            return "✅ Paused"
        return "⚠️ No video player available"

    def _cmd_resume(self, p: dict) -> str:
        return self._cmd_play({})

    def _cmd_filter_action(self, p: dict) -> str:
        name = p.get('name', '').replace('_', ' ').strip().title()
        show = p.get('show', 'true').lower() in ('true', '1', 'yes', 'on')
        scene = self._window.signal_scene
        if name in scene.visible_actions:
            scene.set_action_filter(name, show)
            action = "shown" if show else "hidden"
            return f"✅ Action '{name}' {action}"
        else:
            available = ', '.join(scene.visible_actions.keys())
            return f"⚠️ Action '{name}' not found. Available: {available}"

    def _cmd_filter_object(self, p: dict) -> str:
        name = p.get('name', '').replace('_', ' ').strip().title()
        show = p.get('show', 'true').lower() in ('true', '1', 'yes', 'on')
        scene = self._window.signal_scene
        if name in scene.visible_objects:
            scene.set_object_filter(name, show)
            action = "shown" if show else "hidden"
            return f"✅ Object '{name}' {action}"
        else:
            available = ', '.join(scene.visible_objects.keys())
            return f"⚠️ Object '{name}' not found. Available: {available}"

    def _cmd_show_all_filters(self, p: dict) -> str:
        self._window.show_all_filters()
        return "✅ All filters reset — showing everything"

    def _cmd_confidence(self, p: dict) -> str:
        min_c = float(p.get('min', 0.0))
        max_c = float(p.get('max', 1.0))
        self._window.signal_scene.set_confidence_filter(min_c, max_c)
        return f"✅ Confidence filter set: {min_c:.2f} - {max_c:.2f}"

    def _cmd_zoom(self, p: dict) -> str:
        level = int(p.get('level', 50))
        level = max(10, min(200, level))
        self._window.signal_scene.set_zoom(level)
        return f"✅ Zoom set to {level}"

    def _cmd_list_clips(self, p: dict) -> str:
        clips = self._window.edit_scene.clips
        if not clips:
            return "ℹ️ Edit timeline is empty"
        lines = [f"ℹ️ {len(clips)} clips on edit timeline:"]
        for i, (s, e) in enumerate(clips):
            lines.append(f"  {i+1}. {s:.1f}s - {e:.1f}s ({e-s:.1f}s)")
        total = sum(e - s for s, e in clips)
        lines.append(f"  Total: {total:.1f}s")
        return "\n".join(lines)

    def _cmd_save(self, p: dict) -> str:
        if hasattr(self._window.edit_scene, 'save_clips_to_cache'):
            ok = self._window.edit_scene.save_clips_to_cache()
            return "✅ Timeline saved to cache" if ok else "⚠️ Save failed"
        return "⚠️ Cache saving not available"

    def _cmd_export(self, p: dict) -> str:
        fmt = p.get('format', 'edl').lower()
        clips = self._window.edit_scene.clips
        if not clips:
            return "⚠️ No clips to export"
        try:
            # Use the window's exporter
            from signal_timeline_viewer import TimelineExporter
            if fmt in ('edl',):
                result = TimelineExporter.to_edl(clips, self._window.video_path)
                return f"✅ EDL exported: {result}"
            elif fmt in ('xml', 'fcpxml'):
                result = TimelineExporter.to_fcp_xml(clips, self._window.video_path)
                return f"✅ XML exported: {result}"
            else:
                return f"⚠️ Unknown format: {fmt}. Use 'edl' or 'xml'"
        except Exception as e:
            return f"❌ Export failed: {e}"