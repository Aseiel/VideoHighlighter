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
  | { type: "finished"; output: string; outputs?: string[] }
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
