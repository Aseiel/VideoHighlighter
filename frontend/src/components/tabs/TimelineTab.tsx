// The cut list as something you can see and edit.
//
// An automatic run produces a film and a `film.edl.yaml` describing it. That
// file is the point: the note you have after watching a first pass is never
// "score action higher", it is "that clip should start two seconds later, and
// lose the one after it". This tab is that file with a picture attached.
//
// The picture matters specifically because of the music. Clip blocks and the
// beat grid share one time axis, so whether a cut lands on a beat is something
// you can see rather than something you infer from a BPM readout — and when it
// does not, Quantise is right there.

import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import {
  AlertTriangle,
  ArrowDown,
  ArrowUp,
  Copy,
  FileText,
  FolderOpen,
  Music,
  Play,
  Save,
  Scissors,
  Trash2,
  Wand2,
} from "lucide-react"

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
  analyzeMusic,
  getEdl,
  listTransitions,
  renderEdl,
  saveEdl,
  type EdlCut,
  type MusicAnalysisResult,
} from "@/lib/api"
import { basename, pickAudioFile, pickDirectory } from "@/lib/files"
import { formatShort, formatTime, parseTime } from "@/lib/time"

/** Colours cycle so neighbouring clips are always distinguishable; the exact
 *  hues carry no meaning. */
const BLOCK_TINTS = [
  "bg-sky-500/25 border-sky-400/60",
  "bg-emerald-500/25 border-emerald-400/60",
  "bg-amber-500/25 border-amber-400/60",
  "bg-fuchsia-500/25 border-fuchsia-400/60",
  "bg-rose-500/25 border-rose-400/60",
]

const RESOLUTIONS = [
  { label: "Source", width: 0, height: 0 },
  { label: "1080p", width: 1920, height: 1080 },
  { label: "1440p", width: 2560, height: 1440 },
  { label: "4K", width: 3840, height: 2160 },
]

interface Props {
  running: boolean
  onCancel: () => void
  /** Cut list the last automatic run wrote, if any. */
  suggestedPath?: string
}

export function TimelineTab({ running, onCancel, suggestedPath }: Props) {
  const [path, setPath] = useState("")
  const [title, setTitle] = useState("Untitled")
  const [cuts, setCuts] = useState<EdlCut[]>([])
  const [music, setMusic] = useState("")
  const [musicMode, setMusicMode] = useState("replace")
  const [musicVolume, setMusicVolume] = useState(0.8)
  const [resolution, setResolution] = useState(0)
  const [crf, setCrf] = useState(20)
  const [warnings, setWarnings] = useState<string[]>([])
  const [error, setError] = useState("")
  const [status, setStatus] = useState("")
  const [analysis, setAnalysis] = useState<MusicAnalysisResult | null>(null)
  const [kinds, setKinds] = useState<string[]>(["cut", "crossfade"])
  const [selected, setSelected] = useState<number | null>(null)
  const loadedFor = useRef("")

  useEffect(() => {
    void listTransitions().then((r) => {
      if (r.ok && r.transitions?.length) setKinds(r.transitions)
    })
  }, [])

  const load = useCallback(async (target: string) => {
    if (!target) return
    const doc = await getEdl(target)
    if (!doc.ok) {
      setError(doc.error ?? "Could not read that cut list")
      return
    }
    if (!doc.exists) {
      setError("No cut list at that path yet — run the Auto tab first")
      return
    }
    setError("")
    setPath(target)
    setTitle(doc.title ?? "Untitled")
    setCuts(doc.cuts ?? [])
    setMusic(doc.music ?? "")
    setMusicMode(doc.music_mode ?? "replace")
    setMusicVolume(doc.music_volume ?? 0.8)
    setCrf(doc.crf ?? 20)
    setWarnings(doc.warnings ?? [])
    const match = RESOLUTIONS.findIndex(
      (r) => r.width === (doc.width ?? 0) && r.height === (doc.height ?? 0),
    )
    setResolution(match >= 0 ? match : 0)
    setStatus(`Loaded ${doc.cuts?.length ?? 0} cuts`)
  }, [])

  // Offer the last run's cut list without stamping over an open edit.
  useEffect(() => {
    if (suggestedPath && loadedFor.current !== suggestedPath && !cuts.length) {
      loadedFor.current = suggestedPath
      void load(suggestedPath)
    }
  }, [suggestedPath, cuts.length, load])

  useEffect(() => {
    if (!music) {
      setAnalysis(null)
      return
    }
    let stale = false
    void analyzeMusic(music).then((r) => {
      if (!stale) setAnalysis(r.ok ? r : null)
    })
    return () => {
      stale = true
    }
  }, [music])

  const totals = useMemo(() => {
    const source = cuts.reduce((n, c) => n + Math.max(0, c.end - c.start), 0)
    const overlap = cuts
      .slice(0, -1)
      .reduce((n, c) => n + (c.transition !== "cut" ? c.transition_duration : 0), 0)
    return { source, reel: Math.max(0, source - overlap), overlap }
  }, [cuts])

  const patch = (i: number, change: Partial<EdlCut>) =>
    setCuts((list) => list.map((c, n) => (n === i ? { ...c, ...change } : c)))

  const move = (i: number, by: number) =>
    setCuts((list) => {
      const to = i + by
      if (to < 0 || to >= list.length) return list
      const next = [...list]
      ;[next[i], next[to]] = [next[to], next[i]]
      return next
    })

  const remove = (i: number) => {
    setCuts((list) => list.filter((_, n) => n !== i))
    setSelected(null)
  }

  const duplicate = (i: number) =>
    setCuts((list) => [...list.slice(0, i + 1), { ...list[i] }, ...list.slice(i + 1)])

  const applyToAll = (kind: string, seconds: number) =>
    setCuts((list) =>
      list.map((c, i) =>
        i === list.length - 1
          ? { ...c, transition: "cut" }
          : { ...c, transition: kind, transition_duration: seconds },
      ),
    )

  /** Round every clip to a whole bar, locally — the same arithmetic the engine
   *  does, so the timeline shows the result before committing to a render. */
  const quantise = () => {
    const interval = analysis?.ok ? 60 / (analysis.bpm || 0) : 0
    const meter = analysis?.meter ?? 4
    if (!interval || !Number.isFinite(interval)) {
      setStatus("No tempo to quantise against — pick a music track first")
      return
    }
    const bar = interval * meter
    setCuts((list) =>
      list.map((c) => {
        const bars = Math.max(1, Math.round((c.end - c.start) / bar))
        return { ...c, end: c.start + bars * bar }
      }),
    )
    setStatus(`Rounded every clip to the ${bar.toFixed(2)}s bar`)
  }

  const payload = () => ({
    path,
    title,
    music,
    music_mode: musicMode,
    music_volume: musicVolume,
    width: RESOLUTIONS[resolution].width,
    height: RESOLUTIONS[resolution].height,
    crf,
    cuts,
  })

  const save = async () => {
    const res = await saveEdl(payload())
    if (!res.ok) {
      setError(res.error ?? "Could not save")
      return
    }
    setError("")
    setWarnings(res.warnings ?? [])
    setStatus(`Saved — ${formatShort(res.duration ?? 0)} of film`)
  }

  const render = async () => {
    const output = path.replace(/\.edl\.ya?ml$/i, ".mp4") || "film.mp4"
    const res = await renderEdl({ ...payload(), output })
    if (!res.ok) setError(res.error ?? "Could not start the render")
    else {
      setError("")
      setStatus(`Rendering to ${basename(output)}…`)
    }
  }

  if (!cuts.length) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-sm font-medium">
            <Scissors className="size-4" /> Timeline
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <p className="text-xs text-muted-foreground">
            A cut list is what the Auto tab writes next to your film
            (<code>film.edl.yaml</code>). Open it here to trim the clips, change
            how they join, and render again — without re-running detection.
          </p>
          <div className="flex gap-2">
            <Input
              value={path}
              onChange={(e) => setPath(e.target.value)}
              placeholder="path to a .edl.yaml"
            />
            <Button
              variant="outline"
              size="sm"
              onClick={async () => {
                const dir = await pickDirectory()
                if (dir) setPath(`${dir}\\film.edl.yaml`)
              }}
            >
              <FolderOpen className="size-3.5" /> Browse
            </Button>
            <Button size="sm" onClick={() => void load(path)} disabled={!path}>
              Open
            </Button>
          </div>
          {error && <p className="text-xs text-destructive">{error}</p>}
        </CardContent>
      </Card>
    )
  }

  return (
    <div className="space-y-5">
      <Card>
        <CardHeader className="flex-row items-center justify-between space-y-0">
          <CardTitle className="flex items-center gap-2 text-sm font-medium">
            <Scissors className="size-4" /> {title}
          </CardTitle>
          <div className="flex items-center gap-2">
            <Badge variant="secondary">{cuts.length} cuts</Badge>
            <Badge>{formatShort(totals.reel)}</Badge>
            {totals.overlap > 0.05 && (
              <Badge variant="outline">−{totals.overlap.toFixed(1)}s blended</Badge>
            )}
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <TimelineStrip
            cuts={cuts}
            analysis={analysis}
            selected={selected}
            onSelect={setSelected}
          />

          <div className="flex flex-wrap items-center gap-2">
            <Button variant="outline" size="sm" onClick={quantise}>
              <Wand2 className="size-3.5" /> Quantise to the bar
            </Button>
            <Separator orientation="vertical" className="h-6" />
            <span className="text-xs text-muted-foreground">All joins:</span>
            {["cut", "crossfade", "dip_to_black"].map((k) => (
              <Button
                key={k}
                variant="ghost"
                size="sm"
                onClick={() => applyToAll(k, k === "cut" ? 0 : 0.5)}
              >
                {k.replace(/_/g, " ")}
              </Button>
            ))}
            <div className="ml-auto flex items-center gap-2">
              <Button variant="outline" size="sm" onClick={() => void save()}>
                <Save className="size-3.5" /> Save
              </Button>
              {running ? (
                <Button variant="destructive" size="sm" onClick={onCancel}>
                  Cancel
                </Button>
              ) : (
                <Button size="sm" onClick={() => void render()}>
                  <Play className="size-3.5" /> Render
                </Button>
              )}
            </div>
          </div>

          {status && <p className="text-xs text-muted-foreground">{status}</p>}
          {error && <p className="text-xs text-destructive">{error}</p>}
          {warnings.map((w) => (
            <p key={w} className="flex items-center gap-1.5 text-xs text-amber-500">
              <AlertTriangle className="size-3.5 shrink-0" /> {w}
            </p>
          ))}
        </CardContent>
      </Card>

      <div className="grid min-w-0 gap-5 lg:grid-cols-3 [&>*]:min-w-0">
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-sm font-medium">
              <FileText className="size-4" /> Cuts
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            {cuts.map((cut, i) => (
              <CutRow
                key={`${cut.source}-${i}`}
                cut={cut}
                index={i}
                last={i === cuts.length - 1}
                kinds={kinds}
                active={selected === i}
                onSelect={() => setSelected(i)}
                onPatch={(change) => patch(i, change)}
                onMove={(by) => move(i, by)}
                onRemove={() => remove(i)}
                onDuplicate={() => duplicate(i)}
              />
            ))}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-sm font-medium">
              <Music className="size-4" /> Music &amp; output
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="space-y-1.5">
              <Label className="text-xs">Track</Label>
              <div className="flex gap-2">
                <Input
                  value={music ? basename(music) : ""}
                  readOnly
                  placeholder="none"
                  className="text-xs"
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
              </div>
            </div>

            {analysis?.ok && (
              <div className="flex flex-wrap gap-1.5">
                <Badge>{analysis.bpm?.toFixed(1)} BPM</Badge>
                <Badge variant="secondary">{analysis.meter ?? 4}/4</Badge>
                <Badge variant="outline">
                  bar {(((60 / (analysis.bpm || 1)) * (analysis.meter ?? 4)) || 0).toFixed(2)}s
                </Badge>
              </div>
            )}

            {music && (
              <div className="space-y-1.5">
                <Label className="text-xs">How it sits</Label>
                <Select value={musicMode} onValueChange={setMusicMode}>
                  <SelectTrigger className="w-full">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="replace">Replace the original audio</SelectItem>
                    <SelectItem value="mix">Mix with it</SelectItem>
                    <SelectItem value="duck">Duck under it</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            )}

            <Separator />

            <div className="space-y-1.5">
              <Label className="text-xs">Delivery size</Label>
              <Select
                value={String(resolution)}
                onValueChange={(v) => setResolution(Number(v))}
              >
                <SelectTrigger className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {RESOLUTIONS.map((r, i) => (
                    <SelectItem key={r.label} value={String(i)}>
                      {r.label}
                      {r.width ? ` (${r.width}x${r.height})` : ""}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <p className="text-xs text-muted-foreground">
                Camera footage is often 5.3K, which makes a two-minute reel well
                over a gigabyte.
              </p>
            </div>

            <div className="space-y-1.5">
              <Label className="text-xs">Quality (CRF {crf})</Label>
              <Input
                type="number"
                value={crf}
                min={14}
                max={32}
                onChange={(e) => setCrf(Number(e.target.value) || 20)}
              />
              <p className="text-xs text-muted-foreground">
                Lower is better and bigger. 18 is close to source, 23 is small.
              </p>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

/** Clip blocks and the beat grid on one time axis — so "is this cut on the
 *  beat" is a thing you look at rather than a thing you calculate. */
function TimelineStrip({
  cuts,
  analysis,
  selected,
  onSelect,
}: {
  cuts: EdlCut[]
  analysis: MusicAnalysisResult | null
  selected: number | null
  onSelect: (i: number) => void
}) {
  const spans = useMemo(() => {
    let at = 0
    return cuts.map((c) => {
      const duration = Math.max(0, c.end - c.start)
      const span = { start: at, duration }
      at += duration - (c.transition !== "cut" ? c.transition_duration : 0)
      return span
    })
  }, [cuts])

  const total = spans.length
    ? spans[spans.length - 1].start + spans[spans.length - 1].duration
    : 0
  if (!total) return null

  const pct = (v: number) => `${(v / total) * 100}%`
  const beats = analysis?.ok ? (analysis.beats ?? []) : []
  const downbeats = new Set(analysis?.ok ? (analysis.downbeats ?? []) : [])
  // A tick per beat is unreadable past a few hundred across a screen width.
  const step = Math.max(1, Math.ceil(beats.length / 500))

  return (
    <div className="space-y-1">
      <div className="relative h-14 w-full overflow-hidden rounded-md border bg-muted/20">
        {cuts.map((cut, i) => (
          <button
            key={`${cut.source}-${i}`}
            type="button"
            onClick={() => onSelect(i)}
            title={`${cut.label || basename(cut.source)} — ${formatTime(
              Math.max(0, cut.end - cut.start),
            )}`}
            className={`absolute inset-y-0 border-x transition-opacity hover:opacity-100 ${
              BLOCK_TINTS[i % BLOCK_TINTS.length]
            } ${selected === i ? "opacity-100 ring-2 ring-inset ring-primary" : "opacity-80"}`}
            style={{ left: pct(spans[i].start), width: pct(spans[i].duration) }}
          >
            <span className="pointer-events-none block truncate px-1 text-[10px] leading-[3.5rem]">
              {cut.label || basename(cut.source)}
            </span>
          </button>
        ))}
      </div>

      {beats.length > 0 && (
        <div
          className="relative h-6 w-full overflow-hidden rounded-md border bg-muted/30"
          title="Beat grid — tall ticks are downbeats"
        >
          {(analysis?.sections ?? []).map((s) => (
            <div
              key={`${s.start}-${s.end}`}
              className={`absolute inset-y-0 ${
                s.label === "high"
                  ? "bg-primary/20"
                  : s.label === "mid"
                    ? "bg-primary/10"
                    : ""
              }`}
              style={{ left: pct(s.start), width: pct(s.end - s.start) }}
            />
          ))}
          {beats
            .filter((t, i) => i % step === 0 && t <= total)
            .map((t) => (
              <div
                key={t}
                className={`absolute w-px ${
                  downbeats.has(t) ? "inset-y-0 bg-primary" : "inset-y-2 bg-primary/40"
                }`}
                style={{ left: pct(t) }}
              />
            ))}
          {/* Cut boundaries drawn over the grid: alignment is the whole point. */}
          {spans.slice(1).map((s, i) => (
            <div
              key={`edge-${i}`}
              className="absolute inset-y-0 w-px bg-foreground/70"
              style={{ left: pct(s.start) }}
            />
          ))}
        </div>
      )}

      <div className="flex justify-between text-[10px] text-muted-foreground">
        <span>0:00</span>
        <span>{formatShort(total)}</span>
      </div>
    </div>
  )
}

function CutRow({
  cut,
  index,
  last,
  kinds,
  active,
  onSelect,
  onPatch,
  onMove,
  onRemove,
  onDuplicate,
}: {
  cut: EdlCut
  index: number
  last: boolean
  kinds: string[]
  active: boolean
  onSelect: () => void
  onPatch: (change: Partial<EdlCut>) => void
  onMove: (by: number) => void
  onRemove: () => void
  onDuplicate: () => void
}) {
  // Times are edited as text so a half-typed "1:2" does not snap to something
  // else under the cursor; they commit on blur.
  const [inText, setInText] = useState(formatTime(cut.start))
  const [outText, setOutText] = useState(formatTime(cut.end))

  useEffect(() => {
    setInText(formatTime(cut.start))
    setOutText(formatTime(cut.end))
  }, [cut.start, cut.end])

  const commit = (which: "start" | "end", text: string) => {
    const seconds = parseTime(text)
    if (!Number.isFinite(seconds)) {
      setInText(formatTime(cut.start))
      setOutText(formatTime(cut.end))
      return
    }
    if (which === "start" && seconds < cut.end) onPatch({ start: seconds })
    else if (which === "end" && seconds > cut.start) onPatch({ end: seconds })
    else {
      setInText(formatTime(cut.start))
      setOutText(formatTime(cut.end))
    }
  }

  const duration = Math.max(0, cut.end - cut.start)

  return (
    <div
      onClick={onSelect}
      className={`rounded-md border p-2 transition-colors ${
        active ? "border-primary bg-primary/5" : "hover:border-muted-foreground/40"
      }`}
    >
      <div className="flex items-center gap-2">
        <span className="w-6 shrink-0 text-xs text-muted-foreground">{index + 1}</span>
        <span className="min-w-0 flex-1 truncate text-xs font-medium">
          {cut.label || basename(cut.source)}
        </span>
        <Badge variant="secondary" className="shrink-0">
          {formatTime(duration)}
        </Badge>
        <div className="flex shrink-0">
          <Button variant="ghost" size="sm" onClick={() => onMove(-1)} title="Move up">
            <ArrowUp className="size-3.5" />
          </Button>
          <Button variant="ghost" size="sm" onClick={() => onMove(1)} title="Move down">
            <ArrowDown className="size-3.5" />
          </Button>
          <Button variant="ghost" size="sm" onClick={onDuplicate} title="Duplicate">
            <Copy className="size-3.5" />
          </Button>
          <Button variant="ghost" size="sm" onClick={onRemove} title="Remove">
            <Trash2 className="size-3.5 text-destructive" />
          </Button>
        </div>
      </div>

      <div className="mt-2 flex flex-wrap items-center gap-2">
        <Label className="text-[10px] text-muted-foreground">In</Label>
        <Input
          value={inText}
          onChange={(e) => setInText(e.target.value)}
          onBlur={(e) => commit("start", e.target.value)}
          className="h-7 w-20 font-mono text-xs"
        />
        <Label className="text-[10px] text-muted-foreground">Out</Label>
        <Input
          value={outText}
          onChange={(e) => setOutText(e.target.value)}
          onBlur={(e) => commit("end", e.target.value)}
          className="h-7 w-20 font-mono text-xs"
        />

        {!last && (
          <>
            <Label className="ml-2 text-[10px] text-muted-foreground">Then</Label>
            <Select
              value={cut.transition}
              onValueChange={(v) => onPatch({ transition: v })}
            >
              <SelectTrigger className="h-7 w-36 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {kinds.map((k) => (
                  <SelectItem key={k} value={k}>
                    {k.replace(/_/g, " ")}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {cut.transition !== "cut" && (
              <Input
                type="number"
                step="0.1"
                min="0"
                value={cut.transition_duration}
                onChange={(e) =>
                  onPatch({ transition_duration: Number(e.target.value) || 0 })
                }
                className="h-7 w-16 text-xs"
                title="Transition length in seconds"
              />
            )}
          </>
        )}
      </div>
    </div>
  )
}
