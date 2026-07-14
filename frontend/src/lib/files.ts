// Video file picking. Uses the native Tauri dialog when running inside the app,
// and falls back to nothing outside it (browser dev can't read absolute paths).

export function isTauri(): boolean {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window
}

const VIDEO_EXTS = ["mp4", "mov", "mkv", "avi", "webm", "m4v", "MP4", "MOV"]

export async function pickVideos(): Promise<string[]> {
  if (!isTauri()) {
    // Browser dev: prompt for a path manually so the flow is still testable.
    const manual = window.prompt(
      "Running in browser (no native picker). Paste an absolute video path:",
    )
    return manual ? [manual] : []
  }
  const { open } = await import("@tauri-apps/plugin-dialog")
  const selected = await open({
    multiple: true,
    filters: [{ name: "Video", extensions: VIDEO_EXTS }],
  })
  if (!selected) return []
  return Array.isArray(selected) ? selected : [selected]
}

/** Pick a folder (download destination). Returns null if cancelled. */
export async function pickDirectory(): Promise<string | null> {
  if (!isTauri()) {
    return window.prompt("Paste a destination folder path:") || null
  }
  const { open } = await import("@tauri-apps/plugin-dialog")
  const selected = await open({ directory: true, multiple: false })
  if (!selected) return null
  return Array.isArray(selected) ? (selected[0] ?? null) : selected
}

/** Pick a custom YOLO model (.pt/.onnx), loaded natively by ultralytics. */
export async function pickModelFile(): Promise<string | null> {
  if (!isTauri()) {
    return window.prompt("Paste the path to a .pt/.onnx model:") || null
  }
  const { open } = await import("@tauri-apps/plugin-dialog")
  const selected = await open({
    multiple: false,
    filters: [{ name: "YOLO models", extensions: ["pt", "onnx"] }],
  })
  if (!selected) return null
  return Array.isArray(selected) ? (selected[0] ?? null) : selected
}

export function basename(p: string): string {
  return p.split(/[\\/]/).pop() ?? p
}
