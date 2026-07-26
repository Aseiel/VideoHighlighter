// A preview card for one input video: poster frame, filename, duration, delete.
//
// The poster and duration both come from a hidden <video> pointed at the file
// via Tauri's asset protocol, so there's no sidecar round-trip per card. We seek
// a little way in because frame zero of a real clip is often black.

import { useEffect, useRef, useState } from "react"
import { Trash2, Film } from "lucide-react"
import { basename, fileSrc } from "@/lib/files"
import { acquireDecodeSlot } from "@/lib/decodeGate"

/** Seconds into the clip to grab the poster from. */
const POSTER_AT = 1
/**
 * Widest the poster is ever drawn. Source footage is often 4K; encoding a
 * full-res frame costs megabytes of base64 per card for a ~150px thumbnail.
 */
const POSTER_MAX_W = 320

function formatDuration(secs: number): string {
  if (!Number.isFinite(secs) || secs <= 0) return ""
  const total = Math.round(secs)
  const h = Math.floor(total / 3600)
  const m = Math.floor((total % 3600) / 60)
  const s = total % 60
  const mm = h > 0 ? String(m).padStart(2, "0") : String(m)
  return `${h > 0 ? `${h}:` : ""}${mm}:${String(s).padStart(2, "0")}`
}

export function VideoCard({
  path,
  disabled,
  onRemove,
}: {
  path: string
  disabled?: boolean
  onRemove: () => void
}) {
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const [poster, setPoster] = useState<string | null>(null)
  const [duration, setDuration] = useState(0)
  const [failed, setFailed] = useState(false)
  // The decoder only mounts once a slot is granted; see decodeGate. Without the
  // gate, a folder of 150+ clips mounts 150 <video> decoders at once and takes
  // WebView2 down with it.
  const [decoding, setDecoding] = useState(false)
  const releaseRef = useRef<(() => void) | null>(null)

  const src = fileSrc(path)

  // Reset when the path changes so a stale poster never shows.
  useEffect(() => {
    setPoster(null)
    setDuration(0)
    setFailed(!src)
    setDecoding(false)
  }, [src])

  // Queue for a decode slot; mount the <video> only when one is granted. Release
  // it on unmount so a card scrolled/removed before it finishes doesn't wedge
  // the queue.
  useEffect(() => {
    if (!src) return
    let cancelled = false
    void acquireDecodeSlot().then((release) => {
      if (cancelled) {
        release()
        return
      }
      releaseRef.current = release
      setDecoding(true)
    })
    return () => {
      cancelled = true
      releaseRef.current?.()
      releaseRef.current = null
    }
  }, [src])

  /** Free the decode slot once this card's poster is settled (done or failed). */
  const releaseSlot = () => {
    releaseRef.current?.()
    releaseRef.current = null
  }

  function handleLoadedMetadata() {
    const el = videoRef.current
    if (!el) return
    setDuration(el.duration)
    // Clamp: short clips may be under POSTER_AT entirely.
    el.currentTime = Math.min(POSTER_AT, el.duration / 2 || 0)
  }

  function handleSeeked() {
    const el = videoRef.current
    if (!el || poster) return
    if (!el.videoWidth || !el.videoHeight) return
    const scale = Math.min(1, POSTER_MAX_W / el.videoWidth)
    const canvas = document.createElement("canvas")
    canvas.width = Math.round(el.videoWidth * scale)
    canvas.height = Math.round(el.videoHeight * scale)
    const ctx = canvas.getContext("2d")
    if (!ctx) return
    ctx.drawImage(el, 0, 0, canvas.width, canvas.height)
    try {
      setPoster(canvas.toDataURL("image/jpeg", 0.7))
    } catch (err) {
      // Tainted canvas or unsupported codec — fall back to the icon, but say
      // why: a silent fallback here is indistinguishable from a decode failure.
      console.warn(`poster capture failed for ${path}:`, err)
      setFailed(true)
    } finally {
      // Poster captured (or threw) — hand the decoder slot to the next card.
      releaseSlot()
    }
  }

  return (
    <div className="relative overflow-hidden rounded-md border bg-card">
      <div className="relative flex h-28 items-center justify-center bg-muted">
        {poster ? (
          <img src={poster} alt="" className="size-full object-cover" />
        ) : (
          <Film className="size-6 text-muted-foreground" />
        )}
        {duration > 0 && (
          <span className="absolute right-1 top-1 rounded bg-black/70 px-1 text-[10px] text-white">
            {formatDuration(duration)}
          </span>
        )}
        <button
          aria-label={`Remove ${basename(path)}`}
          className="absolute left-1 top-1 rounded bg-black/70 p-1 text-white transition-colors hover:bg-destructive disabled:opacity-40"
          disabled={disabled}
          onClick={onRemove}
        >
          <Trash2 className="size-3.5" />
        </button>
      </div>
      <p className="line-clamp-2 p-2 text-xs" title={path}>
        {basename(path)}
      </p>

      {/* Hidden decoder that produces the poster + duration. Mounts only once a
          decode slot is granted (decodeGate) and unmounts once the poster is
          captured, so we never hold more than a few decoders open at a time. */}
      {src && decoding && !failed && !poster && (
        <video
          ref={videoRef}
          src={src}
          // The asset protocol sends Access-Control-Allow-Origin, but the
          // webview only honours it on a CORS request. Without this the frame
          // decodes fine yet taints the canvas, and toDataURL throws.
          crossOrigin="anonymous"
          muted
          playsInline
          preload="metadata"
          className="hidden"
          onLoadedMetadata={handleLoadedMetadata}
          onSeeked={handleSeeked}
          onError={() => {
            setFailed(true)
            releaseSlot()
          }}
        />
      )}
    </div>
  )
}
