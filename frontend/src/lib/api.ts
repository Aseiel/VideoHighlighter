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
  | { type: "finished"; output: string }
  | { type: "downloaded"; paths: string[] }
  | { type: "cancelled" }
  | { type: "error"; message: string; traceback?: string }
  | { type: "done" }

export interface HealthResponse {
  status: string
  running: boolean
  run_id: string | null
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
}

/** Identities from the shared face bank (named in the native Timeline Viewer). */
export async function getFaces(): Promise<{
  ok: boolean
  identities: FaceIdentity[]
  error?: string
}> {
  const res = await fetch(`${SIDECAR_BASE}/faces`)
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

/** Label vocabulary for the object/action autocomplete. */
export async function getLabels(
  kind: "objects" | "actions",
): Promise<string[]> {
  try {
    const res = await fetch(`${SIDECAR_BASE}/labels/${kind}`)
    const data = await res.json()
    return data.labels ?? []
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
