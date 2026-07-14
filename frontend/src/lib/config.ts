// Mirrors the gui_config dict that main.py's Worker sends to
// pipeline.run_highlighter. Keep field names identical — the pipeline reads them.

export interface HighlighterConfig {
  scene_points: number
  motion_event_points: number
  motion_peak_points: number
  audio_peak_points: number
  keyword_points: number
  transcript_points: number
  object_points: number
  action_points: number
  clip_time: number
  max_duration: number
  exact_duration: number // 0 = off
  keep_temp: boolean
  skip_highlights: boolean
  highlight_objects: string // comma-separated
  interesting_actions: string // comma-separated
  actions_require_objects: boolean
  auto_min_clip: number
  auto_max_clip: number
  auto_merge_gap: number
  frame_skip: number
  force_reprocess: boolean
}

export const DEFAULT_CONFIG: HighlighterConfig = {
  scene_points: 0,
  motion_event_points: 0,
  motion_peak_points: 0,
  audio_peak_points: 0,
  keyword_points: 0,
  transcript_points: 0,
  object_points: 5,
  action_points: 0,
  clip_time: 0,
  max_duration: 600,
  exact_duration: 0,
  keep_temp: false,
  skip_highlights: false,
  highlight_objects: "person",
  interesting_actions: "",
  actions_require_objects: false,
  auto_min_clip: 2,
  auto_max_clip: 30,
  auto_merge_gap: 2,
  frame_skip: 5,
  force_reprocess: false,
}

/** Convert the form model into the exact dict pipeline.run_highlighter expects. */
export function toGuiConfig(
  c: HighlighterConfig,
  outputBase: string,
): Record<string, unknown> {
  const objects = c.highlight_objects
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean)
  const actions = c.interesting_actions
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean)

  const cfg: Record<string, unknown> = {
    scene_points: c.scene_points,
    motion_event_points: c.motion_event_points,
    motion_peak_points: c.motion_peak_points,
    audio_peak_points: c.audio_peak_points,
    keyword_points: c.keyword_points,
    transcript_points: c.transcript_points,
    beginning_points: 0,
    ending_points: 0,
    object_points: objects.length ? c.object_points : 0,
    action_points: actions.length ? c.action_points : 0,
    clip_time: c.clip_time,
    max_duration: c.max_duration,
    exact_duration: c.exact_duration > 0 ? c.exact_duration : null,
    multi_signal_boost: 1.2,
    min_signals_for_boost: 2,
    keep_temp: c.keep_temp,
    output_file: outputBase || "highlight.mp4",
    highlight_objects: objects.length ? objects : null,
    interesting_actions: actions.length ? actions : null,
    actions_require_objects: c.actions_require_objects,
    skip_highlights: c.skip_highlights,
    frame_skip: c.frame_skip,
    auto_min_clip: c.auto_min_clip,
    auto_max_clip: c.auto_max_clip,
    auto_merge_gap: c.auto_merge_gap,
    use_time_range: false,
    force_reprocess: c.force_reprocess,
  }

  // Drop nulls, matching main.py's behaviour.
  return Object.fromEntries(Object.entries(cfg).filter(([, v]) => v !== null))
}

export function totalPoints(c: HighlighterConfig): number {
  const objects = c.highlight_objects.split(",").filter((s) => s.trim()).length
  const actions = c.interesting_actions.split(",").filter((s) => s.trim()).length
  return (
    c.scene_points +
    c.motion_event_points +
    c.motion_peak_points +
    c.audio_peak_points +
    c.keyword_points +
    c.transcript_points +
    (objects ? c.object_points : 0) +
    (actions ? c.action_points : 0)
  )
}
