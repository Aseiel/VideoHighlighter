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
  yolo_custom_model_path: string
  obj_confidence: number
  // Advanced — actions
  sample_rate: number
  action_backend: string
  r3d_model: string
  action_models: string
  // Advanced — visualization
  draw_object_boxes: boolean
  draw_action_labels: boolean
  // Avoid
  avoid_enabled: boolean
  avoid_method: string
  // Quality gate
  quality_gate: boolean
  quality_threshold: number
  // Reel + music (music_volume is the UI 0-100 scale; divided by 100 for the engine)
  music_path: string
  music_mode: string
  music_volume: number
  combine_reel: boolean
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
  yolo_custom_model_path: "",
  obj_confidence: 30,
  sample_rate: 5,
  action_backend: "auto",
  r3d_model: "r3d_18",
  action_models: "intel_only",
  draw_object_boxes: false,
  draw_action_labels: false,
  avoid_enabled: false,
  avoid_method: "skip",
  quality_gate: false,
  quality_threshold: 60,
  music_path: "",
  music_mode: "replace",
  music_volume: 80,
  combine_reel: true,
}

export const MUSIC_MODES = [
  { value: "replace", label: "Replace clip audio" },
  { value: "mix", label: "Mix under clip audio" },
  { value: "duck", label: "Duck under clip audio" },
]

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
export const ACTION_MODELS = [
  { value: "intel_only", label: "Kinetics-400 pretrained" },
  { value: "custom_only", label: "Custom OpenVINO" },
  { value: "r3d_custom_only", label: "R3D fine-tuned" },
  { value: "mixed", label: "Mixed — all available models" },
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

export interface RunExtras {
  avoidIds?: string[]
  /** Manual avoid ranges [[startSec, endSec], …] set in the Timeline Viewer. */
  avoidRanges?: [number, number][]
  /** Percent range + the first video's duration; applied only when both exist. */
  timeRange?: { enabled: boolean; startPct: number; endPct: number }
  duration?: number
  /** True when this run will be followed by a /combine step. Music is then
   *  applied once to the reel, not baked into every clip, so it's withheld here. */
  willCombine?: boolean
}

/** Convert the form model into the exact dict pipeline.run_highlighter expects. */
export function toGuiConfig(
  c: HighlighterConfig,
  outputBase: string,
  videoPaths: string[] = [],
  extras: RunExtras = {},
): Record<string, unknown> {
  const avoidIds = extras.avoidIds ?? []
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
    yolo_custom_model_path: c.yolo_custom_model_path || null,
    obj_confidence: c.obj_confidence,
    action_models: c.action_models,
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
    avoid_manual_ranges: extras.avoidRanges ?? [],
    face_db_path: "./cache/face_db.json",
    force_reprocess: c.force_reprocess,
    // Quality gate always travels to the pipeline (off by default costs nothing).
    quality_gate: c.quality_gate,
    quality_threshold: c.quality_threshold,
  }

  // Music: when a combine step follows, the reel gets the music once (via
  // /combine), so don't also bake it into every individual clip. Empty path = off.
  if (!extras.willCombine && c.music_path) {
    cfg.music_path = c.music_path
    cfg.music_mode = c.music_mode
    cfg.music_volume = c.music_volume / 100
  }

  // Time range needs a known duration to convert percentages to seconds — the
  // Qt GUI applies it only when the checkbox is on AND duration > 0.
  const tr = extras.timeRange
  const dur = extras.duration ?? 0
  if (tr?.enabled && dur > 0) {
    cfg.use_time_range = true
    cfg.range_start = Math.round((tr.startPct / 100) * dur)
    cfg.range_end = Math.round((tr.endPct / 100) * dur)
  } else {
    cfg.use_time_range = false
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

/**
 * config.yaml -> form state. Section shape mirrors main.py's save_config, so the
 * web UI and the Qt GUI read each other's settings.
 */
export function fromConfigFile(
  raw: Record<string, any>,
): Partial<HighlighterConfig> {
  const s = raw.scoring ?? {}
  const h = raw.highlights ?? {}
  const a = raw.advanced ?? {}
  const o = raw.objects ?? {}
  const act = raw.actions ?? {}
  const t = raw.transcript ?? {}
  const sub = raw.subtitles ?? {}
  const vis = raw.visualization ?? {}
  const av = raw.avoid ?? {}
  const join = (v: unknown) => (Array.isArray(v) ? v.join(", ") : "")
  const pick = <T,>(v: T | undefined, d: T): T => (v === undefined ? d : v)

  return {
    scene_points: pick(s.scene_points, DEFAULT_CONFIG.scene_points),
    motion_event_points: pick(s.motion_event_points, DEFAULT_CONFIG.motion_event_points),
    motion_peak_points: pick(s.motion_peak_points, DEFAULT_CONFIG.motion_peak_points),
    audio_peak_points: pick(s.audio_peak_points, DEFAULT_CONFIG.audio_peak_points),
    keyword_points: pick(s.keyword_points, DEFAULT_CONFIG.keyword_points),
    transcript_points: pick(s.transcript_points, DEFAULT_CONFIG.transcript_points),
    object_points: pick(s.object_points, DEFAULT_CONFIG.object_points),
    action_points: pick(s.action_points, DEFAULT_CONFIG.action_points),
    clip_time: pick(h.clip_time, DEFAULT_CONFIG.clip_time),
    max_duration: pick(h.max_duration, DEFAULT_CONFIG.max_duration),
    exact_duration: pick(h.exact_duration, DEFAULT_CONFIG.exact_duration),
    auto_min_clip: pick(h.auto_min_clip, DEFAULT_CONFIG.auto_min_clip),
    auto_max_clip: pick(h.auto_max_clip, DEFAULT_CONFIG.auto_max_clip),
    auto_merge_gap: pick(h.auto_merge_gap, DEFAULT_CONFIG.auto_merge_gap),
    keep_temp: pick(h.keep_temp, DEFAULT_CONFIG.keep_temp),
    skip_highlights: pick(h.skip_highlights, DEFAULT_CONFIG.skip_highlights),
    highlight_objects: join(o.interesting),
    interesting_actions: join(act.interesting),
    actions_require_objects: pick(act.require_objects, false),
    obj_confidence: pick(o.confidence, DEFAULT_CONFIG.obj_confidence),
    use_transcript: pick(t.enabled, false),
    transcript_model: pick(t.model, DEFAULT_CONFIG.transcript_model),
    transcript_source_lang: pick(t.source_lang, DEFAULT_CONFIG.transcript_source_lang),
    search_keywords: join(t.search_keywords),
    create_subtitles: pick(sub.enabled, false),
    source_lang: pick(sub.source_lang, DEFAULT_CONFIG.source_lang),
    target_lang: pick(sub.target_lang, DEFAULT_CONFIG.target_lang),
    frame_skip: pick(a.frame_skip, DEFAULT_CONFIG.frame_skip),
    vr_mode: pick(a.vr_mode, false),
    object_frame_skip: pick(a.object_frame_skip, DEFAULT_CONFIG.object_frame_skip),
    sample_rate: pick(a.sample_rate, DEFAULT_CONFIG.sample_rate),
    yolo_type: pick(a.yolo_type, DEFAULT_CONFIG.yolo_type),
    yolo_model_size: pick(a.yolo_model_size, DEFAULT_CONFIG.yolo_model_size),
    yolo_custom_model_path: pick(a.yolo_custom_model_path, ""),
    action_backend: pick(a.action_backend, DEFAULT_CONFIG.action_backend),
    r3d_model: pick(a.r3d_model, DEFAULT_CONFIG.r3d_model),
    action_models: pick(a.action_models, DEFAULT_CONFIG.action_models),
    draw_object_boxes: pick(vis.draw_object_boxes, false),
    draw_action_labels: pick(vis.draw_action_labels, false),
    avoid_enabled: pick(av.face_recognition_enabled, false),
    quality_gate: pick((raw.quality ?? {}).gate, DEFAULT_CONFIG.quality_gate),
    quality_threshold: pick((raw.quality ?? {}).threshold, DEFAULT_CONFIG.quality_threshold),
    music_path: pick((raw.music ?? {}).path, DEFAULT_CONFIG.music_path),
    music_mode: pick((raw.music ?? {}).mode, DEFAULT_CONFIG.music_mode),
    music_volume: pick((raw.music ?? {}).volume, DEFAULT_CONFIG.music_volume),
    combine_reel: pick((raw.download ?? {}).auto_combine, DEFAULT_CONFIG.combine_reel),
  }
}

/** Form state -> config.yaml sections, matching main.py's save_config. */
export function toConfigFile(
  c: HighlighterConfig,
  opts: {
    videoPaths: string[]
    output: string
    timeRange: { enabled: boolean; startPct: number; endPct: number }
    download: Record<string, unknown>
  },
): Record<string, unknown> {
  return {
    video: { paths: opts.videoPaths },
    // auto_combine lives in the download section for Qt parity (its "combine all
    // highlights" checkbox); the web reel toggle reuses the same key.
    download: { ...opts.download, auto_combine: c.combine_reel },
    highlights: {
      clip_time: c.clip_time,
      output: opts.output,
      max_duration: c.max_duration,
      exact_duration: c.exact_duration,
      keep_temp: c.keep_temp,
      skip_highlights: c.skip_highlights,
      auto_min_clip: c.auto_min_clip,
      auto_max_clip: c.auto_max_clip,
      auto_merge_gap: c.auto_merge_gap,
      use_time_range: opts.timeRange.enabled,
      range_start_pct: opts.timeRange.startPct,
      range_end_pct: opts.timeRange.endPct,
    },
    scoring: {
      scene_points: c.scene_points,
      motion_event_points: c.motion_event_points,
      motion_peak_points: c.motion_peak_points,
      audio_peak_points: c.audio_peak_points,
      keyword_points: c.keyword_points,
      transcript_points: c.transcript_points,
      object_points: c.object_points,
      action_points: c.action_points,
      multi_signal_boost: 1.2,
      min_signals_for_boost: 2,
    },
    actions: {
      interesting: splitList(c.interesting_actions),
      require_objects: c.actions_require_objects,
    },
    objects: {
      interesting: splitList(c.highlight_objects),
      confidence: c.obj_confidence,
    },
    keywords: {
      transcript_file: "transcript.txt",
      interesting: splitList(c.search_keywords),
    },
    transcript: {
      enabled: c.use_transcript,
      model: c.transcript_model,
      source_lang: c.transcript_source_lang,
      search_keywords: splitList(c.search_keywords),
    },
    subtitles: {
      enabled: c.create_subtitles,
      source_lang: c.source_lang,
      target_lang: c.target_lang,
    },
    advanced: {
      frame_skip: c.frame_skip,
      vr_mode: c.vr_mode,
      object_frame_skip: c.object_frame_skip,
      sample_rate: c.sample_rate,
      yolo_type: c.yolo_type,
      yolo_model_size: c.yolo_model_size,
      yolo_custom_model_path: c.yolo_custom_model_path,
      action_backend: c.action_backend,
      r3d_model: c.r3d_model,
      action_models: c.action_models,
    },
    visualization: {
      draw_object_boxes: c.draw_object_boxes,
      draw_action_labels: c.draw_action_labels,
    },
    avoid: { face_recognition_enabled: c.avoid_enabled },
    quality: { gate: c.quality_gate, threshold: c.quality_threshold },
    // volume persisted on the UI 0-100 scale; divided by 100 only when sent to the engine.
    music: { path: c.music_path, mode: c.music_mode, volume: c.music_volume },
  }
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
