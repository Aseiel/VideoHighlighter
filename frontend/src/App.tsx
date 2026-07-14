import { useEffect, useRef, useState } from "react"
import {
  Film,
  Moon,
  Sun,
  Plus,
  Trash2,
  Play,
  Square,
  Sparkles,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Checkbox } from "@/components/ui/checkbox"
import { Progress } from "@/components/ui/progress"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { Toaster } from "@/components/ui/sonner"
import { toast } from "sonner"
import { NumberField } from "@/components/NumberField"
import { useTheme } from "@/lib/theme"
import { pickVideos, basename } from "@/lib/files"
import { DEFAULT_CONFIG, toGuiConfig, totalPoints } from "@/lib/config"
import type { HighlighterConfig } from "@/lib/config"
import {
  startRun,
  cancelRun,
  openEventSocket,
  getHealth,
  type RunEvent,
} from "@/lib/api"

type LogLine = { text: string; kind: "info" | "err" | "ok" }

export default function App() {
  const { theme, toggle } = useTheme()
  const [videos, setVideos] = useState<string[]>([])
  const [output, setOutput] = useState("highlight.mp4")
  const [cfg, setCfg] = useState<HighlighterConfig>(DEFAULT_CONFIG)
  const [running, setRunning] = useState(false)
  const [progress, setProgress] = useState(0)
  const [task, setTask] = useState("")
  const [log, setLog] = useState<LogLine[]>([])
  const [online, setOnline] = useState<boolean | null>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const logEndRef = useRef<HTMLDivElement | null>(null)

  const set = <K extends keyof HighlighterConfig>(k: K, v: HighlighterConfig[K]) =>
    setCfg((c) => ({ ...c, [k]: v }))

  // Health poll so the UI shows whether the Python engine is reachable.
  useEffect(() => {
    let alive = true
    const check = () =>
      getHealth()
        .then((h) => {
          if (!alive) return
          setOnline(true)
          setRunning(h.running)
        })
        .catch(() => alive && setOnline(false))
    check()
    const id = setInterval(check, 4000)
    return () => {
      alive = false
      clearInterval(id)
    }
  }, [])

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [log])

  const appendLog = (text: string, kind: LogLine["kind"] = "info") =>
    setLog((l) => [...l, { text, kind }])

  const handleEvent = (e: RunEvent) => {
    switch (e.type) {
      case "started":
        appendLog("=== Pipeline started ===", "ok")
        break
      case "log":
        appendLog(e.message)
        break
      case "progress":
        if (e.total > 0) setProgress(Math.round((e.current / e.total) * 100))
        setTask(e.detail ? `${e.task} — ${e.detail}` : e.task)
        break
      case "finished":
        appendLog(`✔ Finished: ${e.output || "(no output)"}`, "ok")
        toast.success("Highlight generation complete")
        break
      case "cancelled":
        appendLog("⏹ Cancelled", "err")
        toast("Run cancelled")
        break
      case "error":
        appendLog(`✖ ${e.message}`, "err")
        toast.error("Pipeline error")
        break
      case "done":
        setRunning(false)
        setTask("")
        wsRef.current?.close()
        wsRef.current = null
        break
    }
  }

  const onRun = async () => {
    if (!videos.length) return toast.error("Add at least one video")
    if (totalPoints(cfg) === 0 && !cfg.skip_highlights)
      return toast.error("Set at least one scoring point")

    setLog([])
    setProgress(0)
    setRunning(true)
    wsRef.current = openEventSocket(handleEvent)
    // Give the socket a tick to connect before the run emits events.
    await new Promise((r) => setTimeout(r, 150))
    const res = await startRun(videos, toGuiConfig(cfg, output))
    if (!res.ok) {
      appendLog(`✖ ${res.error}`, "err")
      toast.error(res.error ?? "Failed to start")
      setRunning(false)
      wsRef.current?.close()
    }
  }

  const onCancel = async () => {
    await cancelRun()
    appendLog("⏹ Cancellation requested…", "err")
  }

  const addVideos = async () => {
    const picked = await pickVideos()
    if (picked.length) setVideos((v) => [...new Set([...v, ...picked])])
  }

  return (
    <div className="mx-auto flex min-h-screen max-w-5xl flex-col gap-5 p-6">
      {/* Header */}
      <header className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="grid size-10 place-items-center rounded-xl bg-primary/15 text-primary">
            <Film className="size-5" />
          </div>
          <div>
            <h1 className="text-lg font-semibold tracking-tight">
              Video Highlighter
            </h1>
            <p className="text-xs text-muted-foreground">
              Engine{" "}
              <span
                className={
                  online === null
                    ? "text-muted-foreground"
                    : online
                    ? "text-[color:var(--success)]"
                    : "text-destructive"
                }
              >
                {online === null ? "…" : online ? "online" : "offline"}
              </span>
            </p>
          </div>
        </div>
        <Button variant="ghost" size="icon" onClick={toggle} title="Toggle theme">
          {theme === "dark" ? <Sun className="size-4" /> : <Moon className="size-4" />}
        </Button>
      </header>

      {/* Input videos */}
      <Card>
        <CardHeader className="flex-row items-center justify-between space-y-0">
          <CardTitle className="text-sm font-medium">Input Videos</CardTitle>
          <div className="flex gap-2">
            <Button size="sm" variant="secondary" onClick={addVideos}>
              <Plus className="size-4" /> Add
            </Button>
            <Button
              size="sm"
              variant="ghost"
              onClick={() => setVideos([])}
              disabled={!videos.length}
            >
              <Trash2 className="size-4" /> Clear
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          {videos.length === 0 ? (
            <p className="rounded-md border border-dashed py-6 text-center text-sm text-muted-foreground">
              No videos added yet
            </p>
          ) : (
            <ul className="space-y-1">
              {videos.map((v) => (
                <li
                  key={v}
                  className="flex items-center justify-between rounded-md bg-muted/50 px-3 py-1.5 text-sm"
                >
                  <span className="truncate" title={v}>
                    {basename(v)}
                  </span>
                  <button
                    className="text-muted-foreground hover:text-destructive"
                    onClick={() => setVideos((l) => l.filter((x) => x !== v))}
                  >
                    <Trash2 className="size-3.5" />
                  </button>
                </li>
              ))}
            </ul>
          )}
          <Separator className="my-4" />
          <div className="grid grid-cols-[auto_1fr] items-center gap-3">
            <Label className="text-sm text-muted-foreground">Output name</Label>
            <Input
              value={output}
              onChange={(e) => setOutput(e.target.value)}
              className="h-8"
            />
          </div>
        </CardContent>
      </Card>

      {/* Settings grid */}
      <div className="grid gap-5 md:grid-cols-2">
        <Card>
          <CardHeader className="flex-row items-center justify-between space-y-0">
            <CardTitle className="text-sm font-medium">Scoring Points</CardTitle>
            <Badge variant={totalPoints(cfg) ? "default" : "secondary"}>
              total {totalPoints(cfg)}
            </Badge>
          </CardHeader>
          <CardContent className="space-y-2.5">
            <NumberField label="Scene" value={cfg.scene_points} onChange={(v) => set("scene_points", v)} />
            <NumberField label="Motion event" value={cfg.motion_event_points} onChange={(v) => set("motion_event_points", v)} />
            <NumberField label="Motion peak" value={cfg.motion_peak_points} onChange={(v) => set("motion_peak_points", v)} />
            <NumberField label="Audio peak" value={cfg.audio_peak_points} onChange={(v) => set("audio_peak_points", v)} />
            <NumberField label="Object" value={cfg.object_points} onChange={(v) => set("object_points", v)} />
            <NumberField label="Action" value={cfg.action_points} onChange={(v) => set("action_points", v)} />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-sm font-medium">Duration &amp; Cutting</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2.5">
            <NumberField label="Max highlight duration" hint="(s)" value={cfg.max_duration} onChange={(v) => set("max_duration", v)} />
            <NumberField label="Exact duration" hint="(0=off)" value={cfg.exact_duration} onChange={(v) => set("exact_duration", v)} />
            <NumberField label="Clip time" hint="(0=auto)" value={cfg.clip_time} onChange={(v) => set("clip_time", v)} />
            <Separator className="my-1" />
            <NumberField label="Auto min clip" hint="(s)" value={cfg.auto_min_clip} step={0.5} onChange={(v) => set("auto_min_clip", v)} />
            <NumberField label="Auto max clip" hint="(s)" value={cfg.auto_max_clip} step={0.5} onChange={(v) => set("auto_max_clip", v)} />
            <NumberField label="Merge gap" hint="(s)" value={cfg.auto_merge_gap} step={0.5} onChange={(v) => set("auto_merge_gap", v)} />
          </CardContent>
        </Card>
      </div>

      {/* Detection targets */}
      <Card>
        <CardHeader>
          <CardTitle className="text-sm font-medium">Detection Targets</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="grid grid-cols-[auto_1fr] items-center gap-3">
            <Label className="text-sm text-muted-foreground">Objects</Label>
            <Input
              value={cfg.highlight_objects}
              onChange={(e) => set("highlight_objects", e.target.value)}
              placeholder="person, sports ball, dog"
              className="h-8"
            />
          </div>
          <div className="grid grid-cols-[auto_1fr] items-center gap-3">
            <Label className="text-sm text-muted-foreground">Actions</Label>
            <Input
              value={cfg.interesting_actions}
              onChange={(e) => set("interesting_actions", e.target.value)}
              placeholder="high jump, high kick, archery"
              className="h-8"
            />
          </div>
          <div className="flex flex-wrap gap-5 pt-1">
            <label className="flex items-center gap-2 text-sm">
              <Checkbox
                checked={cfg.actions_require_objects}
                onCheckedChange={(v) => set("actions_require_objects", Boolean(v))}
              />
              Only score actions when objects detected
            </label>
            <label className="flex items-center gap-2 text-sm">
              <Checkbox
                checked={cfg.keep_temp}
                onCheckedChange={(v) => set("keep_temp", Boolean(v))}
              />
              Keep temp clips
            </label>
            <label className="flex items-center gap-2 text-sm">
              <Checkbox
                checked={cfg.force_reprocess}
                onCheckedChange={(v) => set("force_reprocess", Boolean(v))}
              />
              Force reprocess (ignore cache)
            </label>
          </div>
        </CardContent>
      </Card>

      {/* Run bar */}
      <div className="flex items-center gap-4">
        {running ? (
          <Button variant="destructive" onClick={onCancel} className="gap-2">
            <Square className="size-4" /> Cancel
          </Button>
        ) : (
          <Button
            onClick={onRun}
            disabled={online === false}
            className="gap-2 bg-[color:var(--success)] text-white hover:opacity-90"
          >
            <Sparkles className="size-4" /> Run Highlighter
          </Button>
        )}
        <div className="flex-1">
          <div className="mb-1 flex justify-between text-xs text-muted-foreground">
            <span>{task || (running ? "Working…" : "Idle")}</span>
            <span className="tabular-nums">{progress}%</span>
          </div>
          <Progress value={progress} />
        </div>
      </div>

      {/* Log */}
      <Card className="flex min-h-0 flex-1 flex-col">
        <CardHeader className="py-3">
          <CardTitle className="flex items-center gap-2 text-sm font-medium">
            <Play className="size-3.5" /> Log Output
          </CardTitle>
        </CardHeader>
        <CardContent className="min-h-0 flex-1 p-0">
          <ScrollArea className="h-64 px-4 pb-4">
            <pre className="whitespace-pre-wrap font-mono text-xs leading-relaxed">
              {log.length === 0 ? (
                <span className="text-muted-foreground">
                  Logs from the engine will appear here…
                </span>
              ) : (
                log.map((l, i) => (
                  <div
                    key={i}
                    className={
                      l.kind === "err"
                        ? "text-destructive"
                        : l.kind === "ok"
                        ? "text-[color:var(--success)]"
                        : ""
                    }
                  >
                    {l.text}
                  </div>
                ))
              )}
              <div ref={logEndRef} />
            </pre>
          </ScrollArea>
        </CardContent>
      </Card>

      <Toaster richColors position="bottom-right" />
    </div>
  )
}
