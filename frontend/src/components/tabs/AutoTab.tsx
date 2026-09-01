// Card-to-film in one tab: pick a source, say what the film should contain,
// give it music, press go. Everything here maps onto modules/auto_pipeline.py's
// stages, and the stage strip is the primary feedback — a single percentage bar
// tells you nothing useful about a run that spends forty minutes in one stage.

import { useCallback, useEffect, useState } from "react"
import {
  AlertCircle,
  Check,
  CircleDashed,
  FileText,
  FolderOpen,
  HardDrive,
  Loader2,
  Music,
  Play,
  RefreshCw,
  Scissors,
  SkipForward,
} from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Checkbox } from "@/components/ui/checkbox"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Separator } from "@/components/ui/separator"
import { Textarea } from "@/components/ui/textarea"
import {
  analyzeMusic,
  getAutoJob,
  getScriptExample,
  listGoProCards,
  listTransitions,
  validateScript,
  type AutoJobStage,
  type AutoStageName,
  type AutoStageStatus,
  type GoProCardInfo,
  type MusicAnalysisResult,
  type ScriptCheck,
} from "@/lib/api"
import { basename, pickAudioFile, pickDirectory, pickScriptFile, pickVideos } from "@/lib/files"

const STAGE_LABELS: Record<AutoStageName, string> = {
  ingest: "Copy from card",
  music: "Analyse music",
  highlight: "Find highlights",
  combine: "Build the reel",
  music_mix: "Lay the music",
}

const STAGE_ORDER: AutoStageName[] = [
  "ingest",
  "music",
  "highlight",
  "combine",
  "music_mix",
]

function gb(bytes: number): string {
  return `${(bytes / 1024 ** 3).toFixed(2)} GB`
}

function StageIcon({ status }: { status: AutoStageStatus }) {
  switch (status) {
    case "running":
      return <Loader2 className="size-4 animate-spin text-primary" />
    case "done":
      return <Check className="size-4 text-emerald-500" />
    case "skipped":
      return <SkipForward className="size-4 text-muted-foreground" />
    case "failed":
      return <AlertCircle className="size-4 text-destructive" />
    default:
      return <CircleDashed className="size-4 text-muted-foreground/50" />
  }
}

export interface AutoTabProps {
  running: boolean
  /** Live stage status, fed by the App's `stage` events. */
  stages: Partial<Record<AutoStageName, { status: AutoStageStatus; detail: string }>>
  onStart: (opts: {
    dest_root: string
    card_root?: string
    source_paths?: string[]
    folder_name?: string
    script_path?: string
    music_path?: string
    output_name?: string
    transition?: string
    transition_duration?: number
    quantise?: string
    width?: number
    height?: number
    resume: boolean
  }) => void
  onCancel: () => void
}

const RESOLUTIONS = [
  { label: "Source", width: 0, height: 0 },
  { label: "1080p", width: 1920, height: 1080 },
  { label: "1440p", width: 2560, height: 1440 },
  { label: "4K", width: 3840, height: 2160 },
]

export function AutoTab({ running, stages, onStart, onCancel }: AutoTabProps) {
  const [cards, setCards] = useState<GoProCardInfo[]>([])
  const [scanning, setScanning] = useState(false)
  const [selectedCard, setSelectedCard] = useState<string>("")
  const [manualFiles, setManualFiles] = useState<string[]>([])

  const [destRoot, setDestRoot] = useState("")
  const [folderName, setFolderName] = useState("")
  const [outputName, setOutputName] = useState("film.mp4")
  const [resume, setResume] = useState(true)

  const [scriptPath, setScriptPath] = useState("")
  const [scriptText, setScriptText] = useState("")
  const [scriptCheck, setScriptCheck] = useState<ScriptCheck | null>(null)

  const [musicPath, setMusicPath] = useState("")
  const [music, setMusic] = useState<MusicAnalysisResult | null>(null)
  const [analysing, setAnalysing] = useState(false)

  const [prior, setPrior] = useState<AutoJobStage[] | null>(null)

  const [transition, setTransition] = useState("crossfade")
  const [transitionDuration, setTransitionDuration] = useState(0.5)
  const [quantise, setQuantise] = useState("")
  const [resolution, setResolution] = useState(1)
  const [kinds, setKinds] = useState<string[]>(["cut", "crossfade"])

  useEffect(() => {
    void listTransitions().then((r) => {
      if (r.ok && r.transitions?.length) setKinds(r.transitions)
    })
  }, [])

  const scan = useCallback(async () => {
    setScanning(true)
    try {
      const res = await listGoProCards()
      const found = res.ok ? (res.cards ?? []) : []
      setCards(found)
      // Auto-select a lone card: with one reader and one card, making the user
      // click it adds nothing.
      if (found.length === 1) {
        setSelectedCard(found[0].root)
        setFolderName((f) => f || found[0].suggested_folder)
      }
    } finally {
      setScanning(false)
    }
  }, [])

  useEffect(() => {
    void scan()
  }, [scan])

  // Show what a previous run in this folder already finished, so "Resume" is a
  // visible fact rather than a promise.
  useEffect(() => {
    if (!destRoot) {
      setPrior(null)
      return
    }
    let cancelled = false
    void (async () => {
      const res = await getAutoJob(destRoot)
      if (!cancelled) setPrior(res.ok && res.exists ? (res.stages ?? []) : null)
    })()
    return () => {
      cancelled = true
    }
  }, [destRoot, running])

  const chooseCard = (card: GoProCardInfo) => {
    setSelectedCard(card.root)
    setManualFiles([])
    setFolderName(card.suggested_folder)
  }

  const loadExample = async () => {
    const res = await getScriptExample()
    if (res.ok && res.text) {
      setScriptText(res.text)
      setScriptCheck(null)
    }
  }

  const checkScript = async () => {
    setScriptCheck(await validateScript(scriptText))
  }

  const openScript = async () => {
    const path = await pickScriptFile()
    if (path) {
      setScriptPath(path)
      setScriptText("")
      setScriptCheck(null)
    }
  }

  const chooseMusic = async () => {
    const path = await pickAudioFile()
    if (!path) return
    setMusicPath(path)
    setMusic(null)
    setAnalysing(true)
    try {
      setMusic(await analyzeMusic(path))
    } finally {
      setAnalysing(false)
    }
  }

  const hasSource = Boolean(selectedCard) || manualFiles.length > 0
  const canStart = hasSource && Boolean(destRoot) && !running

  const start = () => {
    onStart({
      dest_root: destRoot,
      card_root: selectedCard || undefined,
      source_paths: selectedCard ? undefined : manualFiles,
      folder_name: folderName,
      script_path: scriptPath,
      music_path: musicPath,
      output_name: outputName || "film.mp4",
      transition,
      transition_duration: transitionDuration,
      quantise,
      width: RESOLUTIONS[resolution].width,
      height: RESOLUTIONS[resolution].height,
      resume,
    })
  }

  const activeStages = STAGE_ORDER.filter(
    (s) =>
      (s !== "ingest" || Boolean(selectedCard)) &&
      (s !== "music" || Boolean(musicPath)) &&
      (s !== "music_mix" || Boolean(musicPath)),
  )

  return (
    <div className="space-y-5">
      <div className="grid min-w-0 gap-5 lg:grid-cols-2 [&>*]:min-w-0">
        {/* ── Source ───────────────────────────────────────────────── */}
        <Card>
          <CardHeader className="flex-row items-center justify-between space-y-0">
            <CardTitle className="flex items-center gap-2 text-sm font-medium">
              <HardDrive className="size-4" /> Source
            </CardTitle>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => void scan()}
              disabled={scanning}
            >
              <RefreshCw className={scanning ? "size-3.5 animate-spin" : "size-3.5"} />
              Rescan
            </Button>
          </CardHeader>
          <CardContent className="space-y-3">
            {cards.length === 0 && (
              <p className="text-xs text-muted-foreground">
                {scanning
                  ? "Looking for camera cards…"
                  : "No camera card found. Insert one and rescan, or pick files below."}
              </p>
            )}

            {cards.map((card) => {
              const active = selectedCard === card.root
              return (
                <button
                  key={card.root}
                  type="button"
                  onClick={() => chooseCard(card)}
                  className={`w-full rounded-md border p-3 text-left transition-colors ${
                    active
                      ? "border-primary bg-primary/5"
                      : "hover:border-muted-foreground/40"
                  }`}
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-sm font-medium">{card.label}</span>
                    <Badge variant={active ? "default" : "secondary"}>{card.root}</Badge>
                  </div>
                  <p className="mt-1 text-xs text-muted-foreground">
                    {card.take_count} take{card.take_count === 1 ? "" : "s"} ·{" "}
                    {card.file_count} file{card.file_count === 1 ? "" : "s"} ·{" "}
                    {gb(card.total_bytes)}
                    {card.chaptered_takes > 0 &&
                      ` · ${card.chaptered_takes} chaptered`}
                  </p>
                  {card.firmware && (
                    <p className="text-xs text-muted-foreground/70">
                      firmware {card.firmware}
                    </p>
                  )}
                </button>
              )
            })}

            <Separator />

            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={async () => {
                  const picked = await pickVideos()
                  if (picked.length) {
                    setManualFiles(picked)
                    setSelectedCard("")
                  }
                }}
              >
                <FolderOpen className="size-3.5" /> Pick files instead
              </Button>
              {manualFiles.length > 0 && (
                <span className="text-xs text-muted-foreground">
                  {manualFiles.length} file{manualFiles.length === 1 ? "" : "s"} selected
                </span>
              )}
            </div>
          </CardContent>
        </Card>

        {/* ── Destination ──────────────────────────────────────────── */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-sm font-medium">
              <FolderOpen className="size-4" /> Destination
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="space-y-1.5">
              <Label className="text-xs">Project folder</Label>
              <div className="flex gap-2">
                <Input
                  value={destRoot}
                  onChange={(e) => setDestRoot(e.target.value)}
                  placeholder="Where the footage and the film go"
                />
                <Button
                  variant="outline"
                  size="sm"
                  onClick={async () => {
                    const dir = await pickDirectory()
                    if (dir) setDestRoot(dir)
                  }}
                >
                  Browse
                </Button>
              </div>
            </div>

            {Boolean(selectedCard) && (
              <div className="space-y-1.5">
                <Label className="text-xs">Subfolder for this card</Label>
                <Input
                  value={folderName}
                  onChange={(e) => setFolderName(e.target.value)}
                  placeholder="dated automatically"
                />
              </div>
            )}

            <div className="space-y-1.5">
              <Label className="text-xs">Film file name</Label>
              <Input
                value={outputName}
                onChange={(e) => setOutputName(e.target.value)}
                placeholder="film.mp4"
              />
            </div>

            <label className="flex items-center gap-2 pt-1">
              <Checkbox
                checked={resume}
                onCheckedChange={(v) => setResume(Boolean(v))}
              />
              <span className="text-xs">Resume — skip work already done here</span>
            </label>

            {prior && (
              <p className="text-xs text-muted-foreground">
                Earlier run found:{" "}
                {prior.filter((s) => s.satisfied).map((s) => STAGE_LABELS[s.name]).join(", ") ||
                  "nothing reusable"}
              </p>
            )}
          </CardContent>
        </Card>

        {/* ── Script ───────────────────────────────────────────────── */}
        <Card>
          <CardHeader className="flex-row items-center justify-between space-y-0">
            <CardTitle className="flex items-center gap-2 text-sm font-medium">
              <FileText className="size-4" /> Script
            </CardTitle>
            <div className="flex gap-1">
              <Button variant="ghost" size="sm" onClick={() => void loadExample()}>
                Template
              </Button>
              <Button variant="ghost" size="sm" onClick={() => void openScript()}>
                Open…
              </Button>
            </div>
          </CardHeader>
          <CardContent className="space-y-3">
            {scriptPath ? (
              <div className="flex items-center justify-between gap-2 rounded-md border p-2">
                <span className="truncate text-xs">{basename(scriptPath)}</span>
                <Button variant="ghost" size="sm" onClick={() => setScriptPath("")}>
                  Clear
                </Button>
              </div>
            ) : (
              <>
                <Textarea
                  value={scriptText}
                  onChange={(e) => setScriptText(e.target.value)}
                  placeholder="Optional. Describe the beats of the film — press Template for a starting point."
                  className="h-44 font-mono text-xs"
                  spellCheck={false}
                />
                <div className="flex items-center gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => void checkScript()}
                    disabled={!scriptText.trim()}
                  >
                    Check
                  </Button>
                  {scriptCheck &&
                    (scriptCheck.ok ? (
                      <span className="text-xs text-emerald-500">
                        {scriptCheck.beats?.length} beat
                        {scriptCheck.beats?.length === 1 ? "" : "s"} ·{" "}
                        {scriptCheck.clip_count} clip
                        {scriptCheck.clip_count === 1 ? "" : "s"} ·{" "}
                        {Math.round(scriptCheck.target_duration ?? 0)}s
                      </span>
                    ) : (
                      <span className="text-xs text-destructive">{scriptCheck.error}</span>
                    ))}
                </div>
                {scriptCheck?.warnings?.map((w) => (
                  <p key={w} className="text-xs text-amber-500">
                    {w}
                  </p>
                ))}
              </>
            )}
          </CardContent>
        </Card>

        {/* ── Music ────────────────────────────────────────────────── */}
        <Card>
          <CardHeader className="flex-row items-center justify-between space-y-0">
            <CardTitle className="flex items-center gap-2 text-sm font-medium">
              <Music className="size-4" /> Music
            </CardTitle>
            <Button variant="ghost" size="sm" onClick={() => void chooseMusic()}>
              Choose…
            </Button>
          </CardHeader>
          <CardContent className="space-y-3">
            {!musicPath && (
              <p className="text-xs text-muted-foreground">
                Optional. A track here is analysed for its beat, and laid over the
                finished film.
              </p>
            )}

            {musicPath && (
              <div className="flex items-center justify-between gap-2 rounded-md border p-2">
                <span className="truncate text-xs">{basename(musicPath)}</span>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => {
                    setMusicPath("")
                    setMusic(null)
                  }}
                >
                  Clear
                </Button>
              </div>
            )}

            {analysing && (
              <p className="flex items-center gap-2 text-xs text-muted-foreground">
                <Loader2 className="size-3.5 animate-spin" /> Analysing…
              </p>
            )}

            {music?.ok && (
              <>
                <div className="flex flex-wrap items-center gap-2">
                  <Badge>{music.bpm?.toFixed(1)} BPM</Badge>
                  <Badge variant="secondary">{music.beats?.length ?? 0} beats</Badge>
                  <Badge variant="secondary">
                    {music.downbeats?.length ?? 0} bars
                  </Badge>
                  <Badge variant="outline">{music.backend}</Badge>
                </div>
                <BeatStrip music={music} />
              </>
            )}

            {music && !music.ok && (
              <p className="text-xs text-destructive">{music.error}</p>
            )}
          </CardContent>
        </Card>
      </div>

      {/* ── Cutting ──────────────────────────────────────────────── */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-sm font-medium">
            <Scissors className="size-4" /> Cutting
          </CardTitle>
        </CardHeader>
        <CardContent className="grid gap-4 md:grid-cols-3">
          <div className="space-y-1.5">
            <Label className="text-xs">How clips join</Label>
            <div className="flex gap-2">
              <Select value={transition} onValueChange={setTransition}>
                <SelectTrigger className="flex-1">
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
              {transition !== "cut" && (
                <Input
                  type="number"
                  step="0.1"
                  min="0"
                  value={transitionDuration}
                  onChange={(e) =>
                    setTransitionDuration(Number(e.target.value) || 0)
                  }
                  className="w-20"
                  title="Seconds"
                />
              )}
            </div>
          </div>

          <div className="space-y-1.5">
            <Label className="text-xs">Cut to the music</Label>
            <Select value={quantise || "off"} onValueChange={(v) => setQuantise(v === "off" ? "" : v)}>
              <SelectTrigger className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="off">Off — cut where the action is</SelectItem>
                <SelectItem value="bar">Every cut on a bar</SelectItem>
                <SelectItem value="beat">Every cut on a beat</SelectItem>
              </SelectContent>
            </Select>
            {quantise && !musicPath && (
              <p className="text-xs text-amber-500">Needs a music track.</p>
            )}
          </div>

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
          </div>
        </CardContent>
      </Card>

      {/* ── Pipeline ───────────────────────────────────────────────── */}
      <Card>
        <CardHeader className="flex-row items-center justify-between space-y-0">
          <CardTitle className="text-sm font-medium">Pipeline</CardTitle>
          <div className="flex gap-2">
            {running ? (
              <Button variant="destructive" size="sm" onClick={onCancel}>
                Cancel
              </Button>
            ) : (
              <Button size="sm" onClick={start} disabled={!canStart}>
                <Play className="size-3.5" /> Make the film
              </Button>
            )}
          </div>
        </CardHeader>
        <CardContent>
          {!hasSource && (
            <p className="text-xs text-muted-foreground">
              Pick a camera card or some files to begin.
            </p>
          )}
          {hasSource && !destRoot && (
            <p className="text-xs text-muted-foreground">
              Choose a project folder to begin.
            </p>
          )}

          <ol className="space-y-2">
            {activeStages.map((name) => {
              const state = stages[name]
              const status: AutoStageStatus = state?.status ?? "pending"
              return (
                <li key={name} className="flex items-center gap-3">
                  <StageIcon status={status} />
                  <span
                    className={
                      status === "pending"
                        ? "text-sm text-muted-foreground"
                        : "text-sm"
                    }
                  >
                    {STAGE_LABELS[name]}
                  </span>
                  {state?.detail && (
                    <span className="truncate text-xs text-muted-foreground">
                      {state.detail}
                    </span>
                  )}
                </li>
              )
            })}
          </ol>
        </CardContent>
      </Card>
    </div>
  )
}

/** Beats drawn along the track's length. Downbeats are taller, which is what
 *  makes the bar structure readable at a glance — a uniform comb of ticks
 *  conveys tempo but not phrasing. */
function BeatStrip({ music }: { music: MusicAnalysisResult }) {
  const duration = music.duration ?? 0
  if (!duration) return null
  const downbeats = new Set(music.downbeats ?? [])
  // A tick per beat is unreadable past a few hundred; sample down instead of
  // drawing a solid block.
  const beats = music.beats ?? []
  const step = Math.max(1, Math.ceil(beats.length / 400))

  return (
    <div className="relative h-10 w-full overflow-hidden rounded-md border bg-muted/30">
      {(music.sections ?? []).map((s) => (
        <div
          key={`${s.start}-${s.end}`}
          className={`absolute inset-y-0 ${
            s.label === "high"
              ? "bg-primary/20"
              : s.label === "mid"
                ? "bg-primary/10"
                : "bg-transparent"
          }`}
          style={{
            left: `${(s.start / duration) * 100}%`,
            width: `${((s.end - s.start) / duration) * 100}%`,
          }}
        />
      ))}
      {beats
        .filter((_, i) => i % step === 0)
        .map((t) => (
          <div
            key={t}
            className={`absolute w-px ${
              downbeats.has(t) ? "inset-y-1 bg-primary" : "inset-y-3 bg-primary/40"
            }`}
            style={{ left: `${(t / duration) * 100}%` }}
          />
        ))}
    </div>
  )
}
