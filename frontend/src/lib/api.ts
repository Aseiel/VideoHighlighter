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
