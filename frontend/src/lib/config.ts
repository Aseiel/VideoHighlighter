// Mirrors the gui_config dict that main.py's Worker sends to
// pipeline.run_highlighter. Keep field names identical — the pipeline reads them.

export interface HighlighterConfig {
  // Scoring
  scene_points: number
  motion_event_points: number
  motion_peak_points: number
  audio_peak_points: number
  keyword_points: number
  transcript_points: number
  object_points: number
  action_points: number
  // Duration & cutting
  clip_time: number
  max_duration: number
  exact_duration: number // 0 = off
  auto_min_clip: number
  auto_max_clip: number
  auto_merge_gap: number
  keep_temp: boolean
  skip_highlights: boolean
  // Detection targets
  highlight_objects: string // comma-separated
  interesting_actions: string // comma-separated
  actions_require_objects: boolean
  force_reprocess: boolean
  // Transcript & subtitles
  use_transcript: boolean
  transcript_model: string
  transcript_source_lang: string
  search_keywords: string // comma-separated
  create_subtitles: boolean
  source_lang: string
  target_lang: string
  // Advanced — motion
  frame_skip: number
  vr_mode: boolean
  // Advanced — objects
  object_frame_skip: number
  yolo_type: string
  yolo_model_size: string
  obj_confidence: number
  // Advanced — actions
  sample_rate: number
  action_backend: string
  r3d_model: string
  // Advanced — visualization
  draw_object_boxes: boolean
  draw_action_labels: boolean
  // Avoid
  avoid_enabled: boolean
  avoid_method: string
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
  auto_min_clip: 2,
  auto_max_clip: 30,
  auto_merge_gap: 2,
  keep_temp: false,
  skip_highlights: false,
  highlight_objects: "person",
  interesting_actions: "",
  actions_require_objects: false,
  force_reprocess: false,
  use_transcript: false,
  transcript_model: "base",
  transcript_source_lang: "en",
  search_keywords: "",
  create_subtitles: false,
  source_lang: "en",
  target_lang: "pl",
  frame_skip: 5,
  vr_mode: false,
  object_frame_skip: 10,
  yolo_type: "standard",
  yolo_model_size: "n",
  obj_confidence: 30,
  sample_rate: 5,
  action_backend: "auto",
  r3d_model: "r3d_18",
  draw_object_boxes: false,
  draw_action_labels: false,
  avoid_enabled: false,
  avoid_method: "skip",
}

// Option lists mirror the Qt combo boxes.
export const WHISPER_MODELS = ["tiny", "base", "small", "medium", "large"]
export const TRANSCRIPT_LANGS = [
  "auto", "en", "pl", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh",
]
export const SUBTITLE_LANGS = [
  "en", "pl", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh",
]
export const YOLO_TYPES = [
  { value: "standard", label: "Standard YOLO (80 objects)" },
  { value: "custom", label: "Custom (my trained model)" },
]
export const YOLO_SIZES = [
  { value: "n", label: "Nano (fastest, lowest accuracy)" },
  { value: "s", label: "Small (fast, good balance)" },
  { value: "m", label: "Medium (balanced)" },
  { value: "l", label: "Large (accurate, slower)" },
  { value: "x", label: "Extra-Large (most accurate, slowest)" },
]
export const ACTION_BACKENDS = [
  { value: "auto", label: "Auto (CUDA / OpenVINO / CPU)" },
  { value: "openvino", label: "OpenVINO (Intel GPU / CPU)" },
  { value: "r3d_cuda", label: "R3D + CUDA (NVIDIA GPU)" },
  { value: "r3d_cpu", label: "R3D + CPU (PyTorch, slow)" },
]
export const R3D_MODELS = [
  { value: "r3d_18", label: "R3D-18 (fastest)" },
  { value: "mc3_18", label: "MC3-18 (mixed convolution)" },
  { value: "r2plus1d_18", label: "R(2+1)D-18 (most accurate)" },
]
export const AVOID_METHODS = [
  { value: "skip", label: "Skip those moments" },
  { value: "crop", label: "Crop them out (experimental)" },
]

/** Directory of a path, using whichever separator the path uses. */
function dirname(p: string): string {
  const i = Math.max(p.lastIndexOf("/"), p.lastIndexOf("\\"))
  return i === -1 ? "" : p.slice(0, i)
}

/**
 * Resolve the output_file the pipeline should write, matching main.py:
 * a single video writes next to its source; multiple videos pass the base name
 * through and the pipeline appends '_highlight' per file.
 */
export function resolveOutputFile(
  videoPaths: string[],
  outputBase: string,
): string {
  const base = outputBase || "highlight.mp4"
  if (videoPaths.length === 1) {
    const dir = dirname(videoPaths[0])
    return dir ? `${dir}/${base}` : base
  }
  return base
}

const splitList = (s: string) =>
  s.split(",").map((x) => x.trim()).filter(Boolean)

/** Convert the form model into the exact dict pipeline.run_highlighter expects. */
export function toGuiConfig(
  c: HighlighterConfig,
  outputBase: string,
  videoPaths: string[] = [],
  avoidIds: string[] = [],
): Record<string, unknown> {
  const objects = splitList(c.highlight_objects)
  const actions = splitList(c.interesting_actions)

  const cfg: Record<string, unknown> = {
    scene_points: c.scene_points,
    motion_event_points: c.motion_event_points,
    motion_peak_points: c.motion_peak_points,
    audio_peak_points: c.audio_peak_points,
    // Keyword/transcript points only count when transcript runs (matches Qt).
    keyword_points: c.use_transcript ? c.keyword_points : 0,
    transcript_points: c.use_transcript ? c.transcript_points : 0,
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
    output_file: resolveOutputFile(videoPaths, outputBase),
    highlight_objects: objects.length ? objects : null,
    interesting_actions: actions.length ? actions : null,
    actions_require_objects: c.actions_require_objects,
    use_transcript: c.use_transcript,
    transcript_model: c.transcript_model,
    transcript_source_lang: c.transcript_source_lang,
    search_keywords: c.use_transcript ? splitList(c.search_keywords) : [],
    create_subtitles: c.create_subtitles && c.use_transcript,
    source_lang: c.source_lang,
    target_lang: c.target_lang,
    skip_highlights: c.skip_highlights,
    frame_skip: c.frame_skip,
    vr_mode: c.vr_mode,
    object_frame_skip: c.object_frame_skip,
    yolo_type: c.yolo_type,
    yolo_model_size: c.yolo_model_size,
    sample_rate: c.sample_rate,
    auto_min_clip: c.auto_min_clip,
    auto_max_clip: c.auto_max_clip,
    auto_merge_gap: c.auto_merge_gap,
    draw_object_boxes: c.draw_object_boxes,
    draw_action_labels: c.draw_action_labels,
    action_backend: c.action_backend,
    r3d_model: c.r3d_model,
    avoid_enabled: c.avoid_enabled && avoidIds.length > 0,
    avoid_method: c.avoid_method,
    avoid_identity_ids: avoidIds,
    face_db_path: "./cache/face_db.json",
    use_time_range: false,
    force_reprocess: c.force_reprocess,
  }

  // Skip-highlights zeroes the scoring/duration knobs, matching main.py.
  if (cfg.skip_highlights) {
    Object.assign(cfg, {
      scene_points: 0,
      motion_event_points: 0,
      motion_peak_points: 0,
      audio_peak_points: 0,
      object_points: 0,
      action_points: 0,
      keyword_points: 0,
      clip_time: 0,
      max_duration: 0,
      exact_duration: null,
    })
  }

  // Drop nulls, matching main.py's behaviour.
  return Object.fromEntries(Object.entries(cfg).filter(([, v]) => v !== null))
}

export function totalPoints(c: HighlighterConfig): number {
  const objects = splitList(c.highlight_objects).length
  const actions = splitList(c.interesting_actions).length
  return (
    c.scene_points +
    c.motion_event_points +
    c.motion_peak_points +
    c.audio_peak_points +
    (c.use_transcript ? c.keyword_points + c.transcript_points : 0) +
    (objects ? c.object_points : 0) +
    (actions ? c.action_points : 0)
  )
}
