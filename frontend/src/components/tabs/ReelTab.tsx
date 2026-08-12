// Make a short-form reel from footage that has already been analysed.
//
// The Auto tab produces a film: every good moment, at whatever length they
// came out. A reel is a different object — fifteen to thirty seconds, a cut
// every second or two, and a shape (hook, context, escalation, payoff) that
// the highest-scoring seconds do not have on their own.
//
// So this tab asks for three things only: how long, how fast, and what the
// opening line says. Everything else is derived, and the plan is shown before
// anything is rendered, because the useful moment to discover that 24 seconds
// of calm pacing is impossible is before the encode rather than after it.

import { useCallback, useEffect, useMemo, useState } from "react"
import { Clapperboard, Loader2, Music, Play, RefreshCw, Type } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Separator } from "@/components/ui/separator"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import {
  getReelOptions,
  planReel,
  renderReel,
  type ReelPace,
  type ReelPlan,
  type ReelRequest,
} from "@/lib/api"
import { basename, pickAudioFile, pickDirectory } from "@/lib/files"

const SHAPES = [
  { key: "vertical", label: "Vertical 9:16", width: 1080, height: 1920, fill: "crop" },
  { key: "square", label: "Square 1:1", width: 1080, height: 1080, fill: "crop" },
  { key: "wide", label: "Wide 16:9", width: 1920, height: 1080, fill: "pad" },
]

// The four sections a viewer actually experiences, and what each is for. Only
// the hook is filled in by most people, which is why it is first and explained.
const TEXT_FIELDS = [
  {
    section: "Hook",
    label: "Opening line",
    hint: "The first two seconds. Most people watch muted, so this is what they read.",
    placeholder: "I nearly quit at mile 38",
  },
  {
    section: "Context",
    label: "Context",
    hint: "Where, what, what is at stake. Keep it to a few words.",
    placeholder: "50 miles. Heavy rain. No backup plan.",
  },
  {
    section: "Payoff",
    label: "Ending",
    hint: "The result, the lesson, or a question worth answering.",
    placeholder: "The answer was slowing down",
  },
]

interface Props {
  running: boolean
  onCancel: () => void
  /** Project folder the last automatic run used, if any. */
  suggestedRoot?: string
}

export function ReelTab({ running, onCancel, suggestedRoot }: Props) {
  const [root, setRoot] = useState("")
  const [duration, setDuration] = useState(24)
  const [pace, setPace] = useState("energetic")
  const [shape, setShape] = useState(0)
  const [music, setMusic] = useState("")
  const [texts, setTexts] = useState<Record<string, string>>({})
  const [paces, setPaces] = useState<ReelPace[]>([])
  const [lengths, setLengths] = useState<{ seconds: number; reason: string }[]>([])
  const [plan, setPlan] = useState<ReelPlan | null>(null)
  const [planning, setPlanning] = useState(false)
  const [status, setStatus] = useState("")

  useEffect(() => {
    void getReelOptions().then((r) => {
      if (!r.ok) return
      setPaces(r.paces ?? [])
      setLengths(r.lengths ?? [])
    })
  }, [])

  useEffect(() => {
    if (suggestedRoot && !root) setRoot(suggestedRoot)
  }, [suggestedRoot, root])

  const band = useMemo(() => paces.find((p) => p.key === pace), [paces, pace])

  const request = useCallback(
    (): ReelRequest => ({
      dest_root: root,
      duration,
      pace,
      title: "Reel",
      music,
      width: SHAPES[shape].width,
      height: SHAPES[shape].height,
      fill: SHAPES[shape].fill,
      texts,
    }),
    [root, duration, pace, shape, music, texts],
  )

  // Re-plan whenever a choice changes. Planning only probes durations, so it is
  // cheap enough to run on every keystroke of the length field.
  useEffect(() => {
    if (!root) {
      setPlan(null)
      return
    }
    let stale = false
    setPlanning(true)
    void planReel(request())
      .then((p) => {
        if (!stale) setPlan(p)
      })
      .finally(() => {
        if (!stale) setPlanning(false)
      })
    return () => {
      stale = true
    }
  }, [request, root])

  const tooShort = Boolean(band && duration < band.minimum_duration - 0.5)

  const render = async () => {
    const res = await renderReel(request())
    if (!res.ok) setStatus(res.error ?? "Could not start the render")
    else setStatus(`Rendering ${res.shots} shots to ${basename(res.output ?? "")}…`)
  }

  return (
    <div className="space-y-5">
      <Card>
        <CardHeader className="flex-row items-center justify-between space-y-0">
          <CardTitle className="flex items-center gap-2 text-sm font-medium">
            <Clapperboard className="size-4" /> Reel
          </CardTitle>
          {plan?.ok && (
            <div className="flex items-center gap-2">
              <Badge>{plan.duration?.toFixed(0)}s</Badge>
              <Badge variant="secondary">{plan.shots} shots</Badge>
              <Badge variant="outline">{plan.cuts_per_minute} cuts/min</Badge>
            </div>
          )}
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-1.5">
            <Label className="text-xs">Footage</Label>
            <div className="flex gap-2">
              <Input
                value={root}
                onChange={(e) => setRoot(e.target.value)}
                placeholder="the project folder the Auto tab wrote to"
              />
              <Button
                variant="outline"
                size="sm"
                onClick={async () => {
                  const dir = await pickDirectory()
                  if (dir) setRoot(dir)
                }}
              >
                Browse
              </Button>
            </div>
            <p className="text-xs text-muted-foreground">
              Uses the highlights an earlier run already found, so nothing is
              analysed twice.
            </p>
          </div>

          <Separator />

          <div className="space-y-2">
            <Label className="text-xs">How long</Label>
            <div className="flex flex-wrap items-center gap-2">
              {lengths.map((l) => (
                <Button
                  key={l.seconds}
                  variant={duration === l.seconds ? "default" : "outline"}
                  size="sm"
                  onClick={() => setDuration(l.seconds)}
                  title={l.reason}
                >
                  {l.seconds}s
                </Button>
              ))}
              <Input
                type="number"
                min={5}
                max={180}
                value={duration}
                onChange={(e) => setDuration(Number(e.target.value) || 24)}
                className="w-20"
              />
              <span className="text-xs text-muted-foreground">
                {lengths.find((l) => l.seconds === duration)?.reason ?? "seconds"}
              </span>
            </div>
          </div>

          <div className="grid gap-4 md:grid-cols-2">
            <div className="space-y-1.5">
              <Label className="text-xs">How fast</Label>
              <Select value={pace} onValueChange={setPace}>
                <SelectTrigger className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {paces.map((p) => (
                    <SelectItem key={p.key} value={p.key}>
                      {p.label} — {p.min_shot}–{p.max_shot}s shots
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              {band && (
                <p className="text-xs text-muted-foreground">
                  {band.cuts_per_minute[0]}–{band.cuts_per_minute[1]} cuts per
                  minute.
                </p>
              )}
              {tooShort && band && (
                <p className="text-xs text-amber-500">
                  A {band.label.toLowerCase()} story needs about{" "}
                  {band.minimum_duration.toFixed(0)}s. Pick a faster pace or a
                  longer reel.
                </p>
              )}
            </div>

            <div className="space-y-1.5">
              <Label className="text-xs">Shape</Label>
              <Select value={String(shape)} onValueChange={(v) => setShape(Number(v))}>
                <SelectTrigger className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {SHAPES.map((s, i) => (
                    <SelectItem key={s.key} value={String(i)}>
                      {s.label} ({s.width}×{s.height})
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <p className="text-xs text-muted-foreground">
                {SHAPES[shape].fill === "crop"
                  ? "Fills the frame — the sides of a wide shot are cropped away."
                  : "Keeps the whole frame, with bars where it does not fit."}
              </p>
            </div>
          </div>

          <div className="space-y-1.5">
            <Label className="flex items-center gap-1.5 text-xs">
              <Music className="size-3.5" /> Music
            </Label>
            <div className="flex gap-2">
              <Input
                value={music ? basename(music) : ""}
                readOnly
                placeholder="optional — shots snap to its beat"
              />
              <Button
                variant="outline"
                size="sm"
                onClick={async () => {
                  const p = await pickAudioFile()
                  if (p) setMusic(p)
                }}
              >
                Pick
              </Button>
              {music && (
                <Button variant="ghost" size="sm" onClick={() => setMusic("")}>
                  Clear
                </Button>
              )}
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="grid min-w-0 gap-5 lg:grid-cols-2 [&>*]:min-w-0">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-sm font-medium">
              <Type className="size-4" /> Words on screen
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {TEXT_FIELDS.map((field) => (
              <div key={field.section} className="space-y-1">
                <Label className="text-xs">{field.label}</Label>
                <Input
                  value={texts[field.section] ?? ""}
                  onChange={(e) =>
                    setTexts((t) => ({ ...t, [field.section]: e.target.value }))
                  }
                  placeholder={field.placeholder}
                />
                <p className="text-xs text-muted-foreground">{field.hint}</p>
              </div>
            ))}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex-row items-center justify-between space-y-0">
            <CardTitle className="text-sm font-medium">The plan</CardTitle>
            {planning && <Loader2 className="size-4 animate-spin text-muted-foreground" />}
          </CardHeader>
          <CardContent className="space-y-3">
            {!root && (
              <p className="text-xs text-muted-foreground">
                Choose a footage folder to see the shot plan.
              </p>
            )}
            {plan && !plan.ok && (
              <p className="text-xs text-destructive">{plan.error}</p>
            )}
            {plan?.ok && (
              <>
                <pre className="overflow-x-auto rounded-md border bg-muted/30 p-2 font-mono text-xs leading-relaxed">
                  {plan.summary}
                </pre>
                <ShotStrip plan={plan} />
              </>
            )}

            <div className="flex items-center gap-2 pt-1">
              {running ? (
                <Button variant="destructive" size="sm" onClick={onCancel}>
                  Cancel
                </Button>
              ) : (
                <Button
                  size="sm"
                  onClick={() => void render()}
                  disabled={!plan?.ok}
                >
                  <Play className="size-3.5" /> Make the reel
                </Button>
              )}
              <Button
                variant="ghost"
                size="sm"
                onClick={() => void planReel(request()).then(setPlan)}
                disabled={!root}
              >
                <RefreshCw className="size-3.5" /> Re-plan
              </Button>
            </div>
            {status && <p className="text-xs text-muted-foreground">{status}</p>}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

/** The shots as proportional blocks, coloured by section. Reading the four
 *  sections as four bands is the quickest way to see whether the shape is
 *  right — a payoff the same width as the body is the usual mistake. */
function ShotStrip({ plan }: { plan: ReelPlan }) {
  const cuts = plan.cuts ?? []
  const total = plan.duration ?? 0
  if (!cuts.length || !total) return null

  const tint: Record<string, string> = {
    Hook: "bg-rose-500/50",
    Context: "bg-amber-500/50",
    Escalation: "bg-sky-500/50",
    Payoff: "bg-emerald-500/50",
  }

  return (
    <div className="space-y-1">
      <div className="flex h-8 w-full overflow-hidden rounded-md border">
        {cuts.map((cut, i) => (
          <div
            key={`${cut.source}-${i}`}
            className={`${tint[cut.label ?? ""] ?? "bg-muted"} border-r border-background/40`}
            style={{ width: `${((cut.duration ?? 0) / total) * 100}%` }}
            title={`${cut.label} — ${(cut.duration ?? 0).toFixed(1)}s${
              cut.text ? `\n"${cut.text}"` : ""
            }`}
          />
        ))}
      </div>
      <div className="flex flex-wrap gap-2">
        {Object.entries(tint).map(([name, cls]) => (
          <span key={name} className="flex items-center gap-1 text-[10px] text-muted-foreground">
            <span className={`inline-block size-2 rounded-sm ${cls}`} /> {name}
          </span>
        ))}
      </div>
    </div>
  )
}
