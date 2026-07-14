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

export function basename(p: string): string {
  return p.split(/[\\/]/).pop() ?? p
}
