import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Checkbox } from "@/components/ui/checkbox"
import { Slider } from "@/components/ui/slider"

export interface TimeRangeState {
  enabled: boolean
  startPct: number
  endPct: number
}

export const DEFAULT_TIME_RANGE: TimeRangeState = {
  enabled: false,
  startPct: 0,
  endPct: 100,
}

interface Props {
  state: TimeRangeState
  onChange: (s: TimeRangeState) => void
  /** Duration of the first input video, seconds. 0 when unknown. */
  duration: number
}

export function fmtTime(sec: number): string {
  const m = Math.floor(sec / 60)
  const s = Math.floor(sec % 60)
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`
}

/** Preset math mirrors main.py's set_slider_preset. */
function preset(kind: string, duration: number): [number, number] | null {
  if (!duration) return null
  switch (kind) {
    case "first_5":
      return [0, Math.round((Math.min(300, duration) / duration) * 100)]
    case "last_5":
      return [Math.round((Math.max(0, duration - 300) / duration) * 100), 100]
    case "last_10":
      return [Math.round((Math.max(0, duration - 600) / duration) * 100), 100]
    case "middle":
      return [Math.round(100 / 3), Math.round((2 / 3) * 100)]
    case "full":
      return [0, 100]
    default:
      return null
  }
}

const PRESETS = [
  { key: "first_5", label: "First 5min" },
  { key: "last_5", label: "Last 5min" },
  { key: "last_10", label: "Last 10min" },
  { key: "middle", label: "Middle" },
  { key: "full", label: "Full video" },
]

export function TimeRange({ state, onChange, duration }: Props) {
  const set = (p: Partial<TimeRangeState>) => onChange({ ...state, ...p })
  const startS = (state.startPct / 100) * duration
  const endS = (state.endPct / 100) * duration
  const selPct = Math.max(0, state.endPct - state.startPct)

  return (
    <Card>
      <CardHeader className="py-3">
        <CardTitle className="text-sm font-medium">
          Processing Time Range
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <label className="flex items-center gap-2 text-sm">
          <Checkbox
            checked={state.enabled}
            onCheckedChange={(v) => set({ enabled: Boolean(v) })}
          />
          Process only specific time range
        </label>

        <p className="text-xs italic text-muted-foreground">
          {duration > 0
            ? `Video duration: ${fmtTime(duration)} (${Math.round(duration)}s)`
            : "Set range in percentages — real times load once a video is added."}
        </p>

        <Slider
          min={0}
          max={100}
          step={1}
          value={[state.startPct, state.endPct]}
          onValueChange={([a, b]) =>
            set({ startPct: Math.min(a, b), endPct: Math.max(a, b) })
          }
          disabled={!state.enabled}
        />

        <div className="flex justify-between text-xs font-semibold tabular-nums">
          <span>
            {duration > 0 ? fmtTime(startS) : `${state.startPct}%`} (
            {state.startPct}%)
          </span>
          <span>
            {duration > 0 ? fmtTime(endS) : `${state.endPct}%`} ({state.endPct}%)
          </span>
        </div>

        <p className="text-xs font-semibold text-[color:var(--success)]">
          {state.enabled
            ? duration > 0
              ? `Selection: ${fmtTime(endS - startS)} (${selPct}% of video)`
              : `Selection: ${selPct}% of video`
            : "Selection: Full video"}
        </p>

        <div className="flex flex-wrap items-center gap-2">
          <span className="text-xs text-muted-foreground">Quick presets:</span>
          {PRESETS.map((p) => (
            <Button
              key={p.key}
              size="sm"
              variant="secondary"
              disabled={!state.enabled || !duration}
              onClick={() => {
                const r = preset(p.key, duration)
                if (r) set({ startPct: r[0], endPct: r[1] })
              }}
            >
              {p.label}
            </Button>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
