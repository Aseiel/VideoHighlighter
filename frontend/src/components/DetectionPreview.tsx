import { useEffect, useRef, useState } from "react"
import { Pause, Play, ChevronLeft, ChevronRight, SkipForward } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Slider } from "@/components/ui/slider"

export interface PreviewFrame {
  jpeg: string
  boxes: { name: string; x: number; y: number; w: number; h: number; conf: number }[]
  sec: number
}

/** ~30s of history at the pipeline's ~8fps, matching the Qt deque(maxlen=250). */
const MAX_FRAMES = 250

interface Props {
  frames: PreviewFrame[]
}

const fmt = (s: number) =>
  `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`

export function DetectionPreview({ frames }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const [frozen, setFrozen] = useState(false)
  // -1 = follow live.
  const [index, setIndex] = useState(-1)

  const view = index === -1 ? frames[frames.length - 1] : frames[index]

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !view) return
    const ctx = canvas.getContext("2d")
    if (!ctx) return
    const img = new Image()
    img.onload = () => {
      canvas.width = img.width
      canvas.height = img.height
      ctx.drawImage(img, 0, 0)
      // Boxes arrive normalised (0..1) so they scale with whatever we drew.
      ctx.lineWidth = 2
      ctx.strokeStyle = "rgb(0,230,90)"
      ctx.font = "bold 12px system-ui, sans-serif"
      for (const b of view.boxes) {
        const x = b.x * img.width
        const y = b.y * img.height
        const w = b.w * img.width
        const h = b.h * img.height
        ctx.strokeRect(x, y, w, h)
        const label = `${b.name} ${b.conf.toFixed(2)}`
        const tw = ctx.measureText(label).width
        ctx.fillStyle = "rgba(0,0,0,0.6)"
        ctx.fillRect(x, Math.max(0, y - 16), tw + 8, 16)
        ctx.fillStyle = "rgb(0,230,90)"
        ctx.fillText(label, x + 4, Math.max(11, y - 4))
      }
    }
    img.src = `data:image/jpeg;base64,${view.jpeg}`
  }, [view])

  // Following live means tracking the newest frame as it arrives.
  useEffect(() => {
    if (!frozen) setIndex(-1)
  }, [frames.length, frozen])

  return (
    <Card>
      <CardHeader className="py-3">
        <CardTitle className="text-sm font-medium">
          Live Detection Preview
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="grid min-h-56 place-items-center overflow-hidden rounded-md border bg-[#101018]">
          {view ? (
            <canvas ref={canvasRef} className="max-h-[420px] w-full object-contain" />
          ) : (
            <p className="p-8 text-center text-sm text-[#8890b0]">
              Waiting for the detection stage (objects / actions)…
            </p>
          )}
        </div>

        {view && (
          <p className="text-xs text-muted-foreground">
            t={fmt(view.sec)} • {view.boxes.length} object(s)
            {frozen && index !== -1 && ` • frame ${index + 1}/${frames.length}`}
          </p>
        )}

        <div className="flex items-center gap-2">
          <Button
            size="sm"
            variant={frozen ? "default" : "secondary"}
            onClick={() => {
              const next = !frozen
              setFrozen(next)
              if (next) setIndex(frames.length - 1)
              else setIndex(-1)
            }}
            disabled={!frames.length}
          >
            {frozen ? <Play className="size-4" /> : <Pause className="size-4" />}
            {frozen ? "Live" : "Freeze"}
          </Button>
          <Button
            size="sm"
            variant="ghost"
            disabled={!frozen || index <= 0}
            onClick={() => setIndex((i) => Math.max(0, (i === -1 ? frames.length - 1 : i) - 1))}
          >
            <ChevronLeft className="size-4" />
          </Button>
          <Slider
            min={0}
            max={Math.max(0, frames.length - 1)}
            step={1}
            value={[index === -1 ? Math.max(0, frames.length - 1) : index]}
            onValueChange={([v]) => {
              setFrozen(true)
              setIndex(v)
            }}
            disabled={!frames.length}
            className="flex-1"
          />
          <Button
            size="sm"
            variant="ghost"
            disabled={!frozen || index === -1 || index >= frames.length - 1}
            onClick={() => setIndex((i) => Math.min(frames.length - 1, i + 1))}
          >
            <ChevronRight className="size-4" />
          </Button>
          <Button
            size="sm"
            variant="ghost"
            onClick={() => {
              setFrozen(false)
              setIndex(-1)
            }}
            disabled={!frames.length}
          >
            <SkipForward className="size-4" /> Live
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}

export { MAX_FRAMES }
