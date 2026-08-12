// Timestamps, matching modules/edl.py's parse_time / format_time.
//
// The two implementations have to agree, because the same string round-trips
// between this editor and the YAML cut list on disk: a value the UI accepts and
// the loader rejects turns a saved edit into an error the user cannot see the
// cause of. The accepted forms are therefore the same four, and nothing else.

/** Seconds from "8", "8.5", "0:08", "1:23.5" or "1:02:03.5". NaN if unreadable. */
export function parseTime(text: string): number {
  const value = (text ?? "").trim()
  if (!/^(?:(\d+):)?(?:(\d+):)?(\d+(?:\.\d+)?)$/.test(value)) return NaN
  const parts = value.split(":")
  const seconds = Number(parts.pop())
  if (!Number.isFinite(seconds)) return NaN
  const minutes = parts.length ? Number(parts.pop()) : 0
  const hours = parts.length ? Number(parts.pop()) : 0
  return hours * 3600 + minutes * 60 + seconds
}

/**
 * "M:SS.sss", or "H:MM:SS.sss" once there is an hour to show, with trailing
 * zeros trimmed.
 *
 * The millisecond precision matters and must match format_time in edl.py. A
 * bar at 66 BPM is 3.63578s; written to one decimal it reads back as 3.6, and
 * that 36ms compounds across twenty cuts into half a bar of drift. These
 * values are typed into a field and saved straight back to the cut list, so
 * rounding here would quietly undo the alignment the engine just computed.
 */
export function formatTime(seconds: number): string {
  const total = Math.max(0, Number(seconds) || 0)
  const hours = Math.floor(total / 3600)
  const minutes = Math.floor((total % 3600) / 60)
  const secs = total % 60
  const pad = (n: number) => String(n).padStart(2, "0")
  let s = secs.toFixed(3).replace(/\.?0+$/, "")
  if (s.split(".")[0].length < 2) s = `0${s}`
  return hours >= 1 ? `${hours}:${pad(minutes)}:${s}` : `${minutes}:${s}`
}

/** Compact form for labels where the tenth is noise ("1:23"). */
export function formatShort(seconds: number): string {
  const total = Math.max(0, Math.round(Number(seconds) || 0))
  const minutes = Math.floor(total / 60)
  return `${minutes}:${String(total % 60).padStart(2, "0")}`
}
