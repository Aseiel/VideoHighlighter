// Client for the Python FastAPI sidecar. In dev the sidecar runs on 8756; in the
// packaged Tauri app it's spawned on the same port. All heavy work happens there;
// this module only sends config and receives log/progress events.

export const SIDECAR_BASE =
  (import.meta.env.VITE_SIDECAR_URL as string | undefined) ??
  "http://127.0.0.1:8756"

const WS_BASE = SIDECAR_BASE.replace(/^http/, "ws")

export type RunEvent =
  | { type: "started"; run_id: string }
  | { type: "log"; message: string }
  | { type: "progress"; current: number; total: number; task: string; detail: string }
  | {
      type: "finished"
      output: string
      outputs?: string[]
      ok?: boolean
      errors?: string[]
      /** Cut list an auto run wrote — what makes the result editable. */
      edl?: string
    }
  | { type: "downloaded"; paths: string[] }
  | { type: "faces_scanned"; count: number }
  /** Detection was skipped because cached results were reused, so no preview
   *  frames will arrive for that stage. */
  | { type: "cache_used"; kind: "object" | "action" }
  | { type: "vision_hit"; timestamp: number; analysis: string }
  | { type: "vision_results"; results: VisionResult[] }
  | {
      type: "preview"
      jpeg: string
      boxes: { name: string; x: number; y: number; w: number; h: number; conf: number }[]
      sec: number
    }
  | { type: "cancelled" }
  | { type: "error"; message: string; traceback?: string }
  | { type: "done" }
  /** Auto-pipeline stage transition. The pipeline is drawn as stages, so this
   *  is a real event rather than something the client parses out of log prose. */
  | { type: "stage"; stage: AutoStageName; status: AutoStageStatus; detail: string }

export interface HealthResponse {
  status: string
  running: boolean
  paused: boolean
  run_id: string | null
}

/** Toggle the live detection preview stream; takes effect mid-run. */
export async function setPreview(enabled: boolean): Promise<{ ok: boolean }> {
  const res = await fetch(`${SIDECAR_BASE}/preview`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ enabled }),
  })
  return res.json()
}

export async function pauseRun(): Promise<{ ok: boolean }> {
  const res = await fetch(`${SIDECAR_BASE}/pause`, { method: "POST" })
  return res.json()
}

export async function resumeRun(): Promise<{ ok: boolean }> {
  const res = await fetch(`${SIDECAR_BASE}/resume`, { method: "POST" })
  return res.json()
}

/** Lifetime analyzed-video counter (same stats file as the Qt GUI). */
export async function getStats(): Promise<{ ok: boolean; analyzed: number }> {
  const res = await fetch(`${SIDECAR_BASE}/stats`)
  return res.json()
}

/** config.yaml — shared with the Qt app. */
export async function getConfigFile(): Promise<{
  ok: boolean
  config: Record<string, any>
}> {
  const res = await fetch(`${SIDECAR_BASE}/config`)
  return res.json()
}

export async function saveConfigFile(
  config: Record<string, unknown>,
): Promise<{ ok: boolean; error?: string }> {
  const res = await fetch(`${SIDECAR_BASE}/config`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ config }),
  })
  return res.json()
}

/** Duration + dimensions/rotation via the engine's ffprobe helper. */
export async function getVideoInfo(
  path: string,
): Promise<{
  ok: boolean
  duration: number
  width?: number
  height?: number
  fps?: number
  rotation?: number
}> {
  const res = await fetch(
    `${SIDECAR_BASE}/video-info?path=${encodeURIComponent(path)}`,
  )
  return res.json()
}

/** List video files in a folder (folder-at-once input). */
export async function scanFolder(
  path: string,
  recursive = true,
): Promise<{ ok: boolean; files: string[]; count: number; error?: string }> {
  const res = await fetch(
    `${SIDECAR_BASE}/scan-folder?path=${encodeURIComponent(path)}&recursive=${
      recursive ? 1 : 0
    }`,
  )
  return res.json()
}

export interface CombineRequest {
  files: string[]
  output: string
  music_path?: string
  music_mode?: string
  music_volume?: number
}

/** Assemble finished highlights into one reel. Emits events on /ws. */
export async function combineVideos(
  req: CombineRequest,
): Promise<{ ok: boolean; run_id?: string; error?: string }> {
  const res = await fetch(`${SIDECAR_BASE}/combine`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  })
  return res.json()
}

export interface CompRule {
  name: string
  label: string
  source: string
  region: string
  min_count: number
  max_count: number
  window_secs: number
  persist_secs: number
}

export async function getCompositionRules(): Promise<{
  ok: boolean
  rules: CompRule[]
}> {
  const res = await fetch(`${SIDECAR_BASE}/composition-rules`)
  return res.json()
}

export async function saveCompositionRules(
  rules: CompRule[],
): Promise<{ ok: boolean; error?: string; events?: number }> {
  const res = await fetch(`${SIDECAR_BASE}/composition-rules`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ rules }),
  })
  return res.json()
}

export async function getHealth(): Promise<HealthResponse> {
  const res = await fetch(`${SIDECAR_BASE}/health`)
  if (!res.ok) throw new Error(`health ${res.status}`)
  return res.json()
}

export async function startRun(
  videoPaths: string[],
  config: Record<string, unknown>,
): Promise<{ ok: boolean; run_id?: string; error?: string }> {
  const res = await fetch(`${SIDECAR_BASE}/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ video_paths: videoPaths, config }),
  })
  return res.json()
}

export async function cancelRun(): Promise<{ ok: boolean }> {
  const res = await fetch(`${SIDECAR_BASE}/cancel`, { method: "POST" })
  return res.json()
}

export interface DownloadOptions {
  url: string
  save_dir: string
  pattern?: string
  download_full: boolean
  time_range_start: number
  time_range_end: number
  concurrent: number
  /** Exact URLs from the picker; when set, no listing scrape happens. */
  video_urls?: string[]
}

export interface ListingEntry {
  url: string
  title?: string
  thumbnail_url?: string
  duration?: string | number
}

/** Scrape a listing page into pickable entries (the Browse & Select grid). */
export async function browseListing(
  url: string,
  useBrowser = "auto",
): Promise<{ ok: boolean; entries: ListingEntry[]; error?: string }> {
  const res = await fetch(`${SIDECAR_BASE}/browse-listing`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url, pattern: "auto", use_browser: useBrowser }),
  })
  return res.json()
}

export interface AboutInfo {
  ok: boolean
  version: string
  edition: string
  support_email: string
  website: string
  discord: string
  repo: string
  log_path: string
}

export async function getAbout(): Promise<AboutInfo> {
  const res = await fetch(`${SIDECAR_BASE}/about`)
  return res.json()
}

/** Show debug.log in the file manager (the file you attach to a bug report). */
export async function revealLog(): Promise<{
  ok: boolean
  path?: string
  error?: string
}> {
  const res = await fetch(`${SIDECAR_BASE}/reveal-log`, { method: "POST" })
  return res.json()
}

/** Show a finished highlight video in the file manager. */
export async function revealOutput(path: string): Promise<{
  ok: boolean
  path?: string
  error?: string
}> {
  const res = await fetch(`${SIDECAR_BASE}/reveal-output`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path }),
  })
  return res.json()
}

/** Scrape + download videos. Emits events on the same socket as a run. */
export async function startDownload(
  opts: DownloadOptions,
): Promise<{ ok: boolean; run_id?: string; error?: string }> {
  const res = await fetch(`${SIDECAR_BASE}/download`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(opts),
  })
  return res.json()
}

export interface FaceIdentity {
  id: string
  name: string
  label: string
  avoid: boolean
  count: number
  thumb: string
}

/** Identities from the shared face bank (named in the native Timeline Viewer). */
export async function getFaces(): Promise<{
  ok: boolean
  identities: FaceIdentity[]
  named?: number
  avoided?: number
  error?: string
}> {
  const res = await fetch(`${SIDECAR_BASE}/faces`)
  return res.json()
}

export async function removeFace(id: string): Promise<{ ok: boolean }> {
  const res = await fetch(`${SIDECAR_BASE}/faces/remove`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id }),
  })
  return res.json()
}

export async function nameFace(
  id: string,
  name: string,
): Promise<{ ok: boolean; merged_into?: string; error?: string }> {
  const res = await fetch(`${SIDECAR_BASE}/faces/name`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id, name }),
  })
  return res.json()
}

export async function clearFaces(
  keepNamed: boolean,
): Promise<{ ok: boolean; remaining?: number; error?: string }> {
  const res = await fetch(`${SIDECAR_BASE}/faces/clear`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ keep_named: keepNamed }),
  })
  return res.json()
}

export async function scanFaces(
  videoPath: string,
): Promise<{ ok: boolean; error?: string }> {
  const res = await fetch(`${SIDECAR_BASE}/faces/scan`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ video_path: videoPath }),
  })
  return res.json()
}

export async function setFaceAvoid(
  id: string,
  avoid: boolean,
): Promise<{ ok: boolean; error?: string }> {
  const res = await fetch(`${SIDECAR_BASE}/faces/avoid`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id, avoid }),
  })
  return res.json()
}

/** Object vocabulary; source depends on the configured detector type. */
export async function getObjectLabels(yoloType = "standard"): Promise<string[]> {
  try {
    const res = await fetch(
      `${SIDECAR_BASE}/labels/objects?yolo_type=${encodeURIComponent(yoloType)}`,
    )
    return (await res.json()).labels ?? []
  } catch {
    return []
  }
}

/** Action vocabulary; depends on backend + model selection. */
export async function getActionLabels(
  backend = "auto",
  models = "intel_only",
): Promise<string[]> {
  try {
    const res = await fetch(
      `${SIDECAR_BASE}/labels/actions?backend=${encodeURIComponent(
        backend,
      )}&models=${encodeURIComponent(models)}`,
    )
    return (await res.json()).labels ?? []
  } catch {
    return []
  }
}

/** Local LLM backends (Ollama / llama-cpp) and any Ollama models found. */
export async function getLlmBackends(): Promise<{
  ok: boolean
  backends: string[]
  ollama_models: string[]
  error?: string
}> {
  const res = await fetch(`${SIDECAR_BASE}/llm/backends`)
  return res.json()
}

export interface VisionResult {
  timestamp: number
  score: number
  thumb: string
  analysis: string
}

export async function getClipStatus(): Promise<{
  ok: boolean
  available: boolean
  error?: string | null
}> {
  const res = await fetch(`${SIDECAR_BASE}/llm/clip-status`)
  return res.json()
}

export interface VisionSearchOptions {
  video_path: string
  query: string
  mode: "clip" | "clip_llm" | "llm"
  interval: number
  top_k: number
  threshold: number
  clip_device: string
  backend: string
  model: string
}

/** Find moments matching a query. Streams progress/results on the run socket. */
export async function visionSearch(
  opts: VisionSearchOptions,
): Promise<{ ok: boolean; run_id?: string; error?: string }> {
  const res = await fetch(`${SIDECAR_BASE}/llm/vision-search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(opts),
  })
  return res.json()
}

export async function llmChat(
  backend: string,
  model: string,
  message: string,
  videoPath?: string,
): Promise<{ ok: boolean; answer?: string; had_context?: boolean; error?: string }> {
  const res = await fetch(`${SIDECAR_BASE}/llm/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      backend,
      model,
      message,
      video_path: videoPath ?? null,
    }),
  })
  return res.json()
}

/** Manual avoid ranges marked for a video in the Timeline Viewer. */
export async function getAvoidRanges(
  path: string,
): Promise<{ ok: boolean; ranges: [number, number][] }> {
  const res = await fetch(
    `${SIDECAR_BASE}/avoid-ranges?path=${encodeURIComponent(path)}`,
  )
  return res.json()
}

export async function saveAvoidRanges(
  videoPath: string,
  ranges: [number, number][],
): Promise<{ ok: boolean; error?: string }> {
  const res = await fetch(`${SIDECAR_BASE}/avoid-ranges`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ video_path: videoPath, ranges }),
  })
  return res.json()
}

/** Hand off to the native Qt app (Timeline Viewer / realtime editor). */
export async function openEditor(
  videoPath?: string,
): Promise<{ ok: boolean; error?: string }> {
  const res = await fetch(`${SIDECAR_BASE}/open-editor`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ video_path: videoPath ?? null }),
  })
  return res.json()
}

// ── Camera cards, script, music, and the auto pipeline ────────────────────

/** Stage names must match modules/auto_pipeline.py's STAGE_* constants — they
 *  are the on-disk job format, not display strings. */
export type AutoStageName =
  | "ingest"
  | "music"
  | "highlight"
  | "combine"
  | "music_mix"

export type AutoStageStatus =
  | "pending"
  | "running"
  | "done"
  | "failed"
  | "skipped"

export interface GoProCardInfo {
  root: string
  label: string
  camera_type: string
  firmware: string
  file_count: number
  total_bytes: number
  take_count: number
  chaptered_takes: number
  suggested_folder: string
}

/** Camera cards currently mounted, with what is on each. */
export async function listGoProCards(): Promise<{
  ok: boolean
  cards?: GoProCardInfo[]
  error?: string
}> {
  const res = await fetch(`${SIDECAR_BASE}/gopro/cards`)
  return res.json()
}

export async function getScriptExample(): Promise<{
  ok: boolean
  text?: string
  error?: string
}> {
  const res = await fetch(`${SIDECAR_BASE}/script/example`)
  return res.json()
}

export interface ScriptCheck {
  ok: boolean
  error?: string
  title?: string
  beats?: string[]
  clip_count?: number
  target_duration?: number
  music?: string
  warnings?: string[]
}

/** Parse a script without running anything, so the editor can report a typo
 *  now instead of after twenty minutes of detection. */
export async function validateScript(text: string): Promise<ScriptCheck> {
  const res = await fetch(`${SIDECAR_BASE}/script/validate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  })
  return res.json()
}

export interface MusicSection {
  start: number
  end: number
  energy: number
  label: string
}

export interface MusicAnalysisResult {
  ok: boolean
  error?: string
  bpm?: number
  duration?: number
  beats?: number[]
  downbeats?: number[]
  meter?: number
  backend?: string
  sections?: MusicSection[]
}

export async function analyzeMusic(path: string): Promise<MusicAnalysisResult> {
  const res = await fetch(
    `${SIDECAR_BASE}/music/analysis?path=${encodeURIComponent(path)}`,
  )
  return res.json()
}

export interface AutoRunOptions {
  dest_root: string
  card_root?: string
  source_paths?: string[]
  folder_name?: string
  script_path?: string
  music_path?: string
  output_name?: string
  music_mode?: string
  music_volume?: number
  /** How each clip joins the next. See listTransitions(). */
  transition?: string
  transition_duration?: number
  /** >0 sizes the transition from the music's bar instead of seconds. */
  transition_bars?: number
  /** "bar" | "beat" | "" — round clip lengths so cuts land on the music. */
  quantise?: string
  width?: number
  height?: number
  fps?: number
  crf?: number
  resume?: boolean
  verify?: string
  config?: Record<string, unknown>
}

/** Transition names the renderer accepts, so the UI can never offer one it
 *  would refuse. */
export async function listTransitions(): Promise<{
  ok: boolean
  transitions?: string[]
  default_duration?: number
  error?: string
}> {
  const res = await fetch(`${SIDECAR_BASE}/transitions`)
  return res.json()
}

export interface EdlCut {
  source: string
  start: number
  end: number
  duration?: number
  transition: string
  transition_duration: number
  /** How the blend moves across its length. */
  easing?: string
  /** How soft the transition's edge is, 0–1 of its length. */
  feather?: number
  label?: string
  /** Burnt into the picture for this cut — short-form is watched muted, so
   *  the opening line is part of the edit. */
  text?: string
}

export interface EdlDoc {
  ok: boolean
  exists?: boolean
  error?: string
  title?: string
  music?: string
  music_mode?: string
  music_volume?: number
  width?: number
  height?: number
  fps?: number
  crf?: number
  duration?: number
  source_duration?: number
  warnings?: string[]
  cuts?: EdlCut[]
}

/** Read the cut list an automatic run produced, for the timeline. */
export async function getEdl(path: string): Promise<EdlDoc> {
  const res = await fetch(`${SIDECAR_BASE}/edl?path=${encodeURIComponent(path)}`)
  return res.json()
}

export interface EdlPayload {
  path: string
  title?: string
  music?: string
  music_mode?: string
  music_volume?: number
  width?: number
  height?: number
  fps?: number
  crf?: number
  cuts: EdlCut[]
}

export async function saveEdl(payload: EdlPayload): Promise<{
  ok: boolean
  path?: string
  duration?: number
  warnings?: string[]
  error?: string
}> {
  const res = await fetch(`${SIDECAR_BASE}/edl`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  })
  return res.json()
}

/** Render the timeline as a job; progress arrives on the shared socket. */
export async function renderEdl(
  payload: EdlPayload & { output: string },
): Promise<{ ok: boolean; run_id?: string; error?: string }> {
  const res = await fetch(`${SIDECAR_BASE}/edl/render`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  })
  return res.json()
}

/** Start the card-to-film pipeline. Progress arrives on the shared event
 *  socket as `stage` and `progress` events. */
export async function startAuto(
  opts: AutoRunOptions,
): Promise<{ ok: boolean; run_id?: string; error?: string }> {
  const res = await fetch(`${SIDECAR_BASE}/auto/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(opts),
  })
  return res.json()
}

export interface AutoJobStage {
  name: AutoStageName
  status: AutoStageStatus
  detail: string
  error: string
  seconds: number
  satisfied: boolean
}

export interface AutoJobState {
  ok: boolean
  exists?: boolean
  error?: string
  job_id?: string
  created?: string
  clips?: string[]
  highlights?: string[]
  reel?: string
  final?: string
  errors?: string[]
  stages?: AutoJobStage[]
}

/** Saved state for a destination folder — what a resume would skip. */
export async function getAutoJob(root: string): Promise<AutoJobState> {
  const res = await fetch(
    `${SIDECAR_BASE}/auto/job?root=${encodeURIComponent(root)}`,
  )
  return res.json()
}

// ── Reels ─────────────────────────────────────────────────────────────────

export interface ReelPace {
  key: string
  label: string
  min_shot: number
  max_shot: number
  cuts_per_minute: [number, number]
  /** Shortest reel this pace can hold a full story in. */
  minimum_duration: number
}

export interface ReelTransition {
  key: string
  /** Has an edge, so its softness control means something. Fades and slides
   *  do not: there is nothing to feather. */
  maskable: boolean
  /** Mixes the whole frame rather than moving an edge across it. */
  blend: boolean
}

export interface ReelOptions {
  ok: boolean
  error?: string
  paces?: ReelPace[]
  lengths?: { seconds: number; reason: string }[]
  sections?: string[]
  /** Every name the renderer accepts. */
  transitions?: ReelTransition[]
  /** The ones worth offering first. */
  curated?: string[]
  /** Grouped for a menu — wipes apart from shapes, and so on. */
  families?: { name: string; items: ReelTransition[] }[]
  easings?: string[]
  default_feather?: number
}

/** Pace bands and lengths, read from the planner so the UI cannot drift. */
export async function getReelOptions(): Promise<ReelOptions> {
  const res = await fetch(`${SIDECAR_BASE}/reel/options`)
  return res.json()
}

export interface ReelRequest {
  dest_root?: string
  source_paths?: string[]
  duration: number
  pace: string
  title?: string
  music?: string
  transition?: string
  transition_duration?: number
  /** How the blend moves. Linear is what ffmpeg does unaided. */
  easing?: string
  /** How soft the transition's edge is, 0–1 of its length. */
  feather?: number
  /** Let shots start later than frame zero when the camera is still being
   *  placed at the top of a clip. */
  settle?: boolean
  /** Keep the reel off the same view twice: one shot per spot before any is
   *  reused, and never two near-identical pictures. */
  spread?: boolean
  /** Optional GPX file, used to place clips whose own metadata has no GPS. */
  track?: string
  width?: number
  height?: number
  /** "crop" fills the frame (right for vertical), "pad" keeps the whole shot. */
  fill?: string
  texts?: Record<string, string>
  output?: string
}

export interface ReelPlan {
  ok: boolean
  error?: string
  duration?: number
  shots?: number
  cuts_per_minute?: number
  summary?: string
  cuts?: EdlCut[]
  /** How many shots start later than the top of their clip, and how far in
   *  the latest one begins. */
  trimmed?: number
  trimmed_max?: number
  /** How many distinct spots the reel visits, and how many clips were
   *  available to choose from. */
  places?: number
  clips_seen?: number
}

/** Plan without rendering, so the shot list and the true length are visible
 *  before anyone waits on an encode. */
export async function planReel(req: ReelRequest): Promise<ReelPlan> {
  const res = await fetch(`${SIDECAR_BASE}/reel/plan`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  })
  return res.json()
}

export async function renderReel(req: ReelRequest): Promise<{
  ok: boolean
  run_id?: string
  output?: string
  edl?: string
  duration?: number
  shots?: number
  error?: string
}> {
  const res = await fetch(`${SIDECAR_BASE}/reel/render`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  })
  return res.json()
}

/** Opens the events WebSocket. Caller owns the socket lifecycle. */
export function openEventSocket(onEvent: (e: RunEvent) => void): WebSocket {
  const ws = new WebSocket(`${WS_BASE}/ws`)
  ws.onmessage = (msg) => {
    try {
      onEvent(JSON.parse(msg.data) as RunEvent)
    } catch {
      /* ignore malformed frames */
    }
  }
  return ws
}
