import { useEffect, useRef, useState } from "react"
import {
  Film,
  Moon,
  Sun,
  Plus,
  Trash2,
  Play,
  Pause,
  Square,
  Sparkles,
  MonitorPlay,
  TrendingUp,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Checkbox } from "@/components/ui/checkbox"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Progress } from "@/components/ui/progress"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Separator } from "@/components/ui/separator"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Toaster } from "@/components/ui/sonner"
import { toast } from "sonner"
import { useTheme } from "@/lib/theme"
import { pickVideos, basename } from "@/lib/files"
import {
  DEFAULT_CONFIG,
  toGuiConfig,
  totalPoints,
  fromConfigFile,
  toConfigFile,
} from "@/lib/config"
import type { HighlighterConfig } from "@/lib/config"
import {
  startRun,
  startDownload,
  cancelRun,
  pauseRun,
  resumeRun,
  openEventSocket,
  getHealth,
  getStats,
  getObjectLabels,
  getActionLabels,
  getConfigFile,
  saveConfigFile,
  getVideoInfo,
  getAvoidRanges,
  openEditor,
  type RunEvent,
} from "@/lib/api"
import { TimeRange, DEFAULT_TIME_RANGE, type TimeRangeState } from "@/components/TimeRange"
import {
  DetectionPreview,
  MAX_FRAMES,
  type PreviewFrame,
} from "@/components/DetectionPreview"
import { setPreview } from "@/lib/api"
import { BasicTab } from "@/components/tabs/BasicTab"
import { TranscriptTab } from "@/components/tabs/TranscriptTab"
import { AdvancedTab } from "@/components/tabs/AdvancedTab"
import { AvoidTab } from "@/components/tabs/AvoidTab"
import { LlmChatTab } from "@/components/tabs/LlmChatTab"
import { VisionSearchTab } from "@/components/tabs/VisionSearchTab"
import { AboutTab } from "@/components/tabs/AboutTab"
import type { VisionResult } from "@/lib/api"
import {
  DownloadTab,
  DEFAULT_DOWNLOAD,
  type DownloadSettings,
} from "@/components/tabs/DownloadTab"

type LogLine = { text: string; kind: "info" | "err" | "ok" }

export default function App() {
  const { theme, toggle } = useTheme()
  const [videos, setVideos] = useState<string[]>([])
  const [output, setOutput] = useState("highlight.mp4")
  const [cfg, setCfg] = useState<HighlighterConfig>(DEFAULT_CONFIG)
  const [dl, setDl] = useState<DownloadSettings>(DEFAULT_DOWNLOAD)
  const [avoidIds, setAvoidIds] = useState<string[]>([])
  const [avoidRanges, setAvoidRanges] = useState<[number, number][]>([])
  const [objectLabels, setObjectLabels] = useState<string[]>([])
  const [actionLabels, setActionLabels] = useState<string[]>([])
  const [timeRange, setTimeRange] = useState<TimeRangeState>(DEFAULT_TIME_RANGE)
  const [duration, setDuration] = useState(0)
  const [running, setRunning] = useState(false)
  const [paused, setPaused] = useState(false)
  const [progress, setProgress] = useState(0)
  const [task, setTask] = useState("")
  const [log, setLog] = useState<LogLine[]>([])
  const [online, setOnline] = useState<boolean | null>(null)
  const [analyzed, setAnalyzed] = useState<number | null>(null)
  const [sessionCount, setSessionCount] = useState(0)
  const [faceRefresh, setFaceRefresh] = useState(0)
  const [loaded, setLoaded] = useState(false)
  const [livePreview, setLivePreview] = useState(false)
  const [frames, setFrames] = useState<PreviewFrame[]>([])
  // Shared by the LLM Chat and Visual Search tabs.
  const [llmBackend, setLlmBackend] = useState("")
  const [llmModel, setLlmModel] = useState("")
  const [visionResults, setVisionResults] = useState<VisionResult[]>([])
  const wsRef = useRef<WebSocket | null>(null)
  const logEndRef = useRef<HTMLDivElement | null>(null)
  // Read inside WS callbacks, which close over the mount-time value otherwise.
  const dlRef = useRef(dl)
  dlRef.current = dl

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
          setPaused(h.paused)
        })
        .catch(() => alive && setOnline(false))
    check()
    const id = setInterval(check, 4000)
    return () => {
      alive = false
      clearInterval(id)
    }
  }, [])

  // Restore settings from config.yaml — the same file the Qt GUI reads/writes.
  useEffect(() => {
    void (async () => {
      const res = await getConfigFile()
      if (res.ok) {
        setCfg((c) => ({ ...c, ...fromConfigFile(res.config) }))
        const h = res.config.highlights ?? {}
        setTimeRange({
          enabled: Boolean(h.use_time_range),
          startPct: h.range_start_pct ?? 0,
          endPct: h.range_end_pct ?? 100,
        })
        if (h.output) setOutput(h.output)
        const paths: string[] = res.config.video?.paths ?? []
        if (paths.length) setVideos(paths)
        const d = res.config.download ?? {}
        setDl((s) => ({
          ...s,
          url: d.last_url ?? s.url,
          saveDir: d.save_dir ?? s.saveDir,
          downloadFull: d.download_full ?? s.downloadFull,
          rangeStart: d.time_range_start ?? s.rangeStart,
          rangeEnd: d.time_range_end ?? s.rangeEnd,
          concurrent: d.concurrent_downloads ?? s.concurrent,
          autoAdd: d.auto_add ?? s.autoAdd,
        }))
      }
      setLoaded(true)
    })()
  }, [])

  // Persist settings whenever they settle, mirroring the Qt app's save-on-close.
  // Debounced so typing doesn't thrash the file; gated on `loaded` so we never
  // write defaults over a real config before it has been read.
  useEffect(() => {
    if (!loaded) return
    const id = setTimeout(() => {
      void saveConfigFile(
        toConfigFile(cfg, {
          videoPaths: videos,
          output,
          timeRange,
          download: {
            last_url: dl.url,
            save_dir: dl.saveDir,
            auto_add: dl.autoAdd,
            download_full: dl.downloadFull,
            time_range_start: dl.rangeStart,
            time_range_end: dl.rangeEnd,
            concurrent_downloads: dl.concurrent,
          },
        }),
      )
    }, 800)
    return () => clearTimeout(id)
  }, [cfg, videos, output, timeRange, dl, loaded])

  // Label vocabularies depend on the detector/backend selection, same as Qt.
  useEffect(() => {
    void getObjectLabels(cfg.yolo_type).then(setObjectLabels)
  }, [cfg.yolo_type])

  useEffect(() => {
    void getActionLabels(cfg.action_backend, cfg.action_models).then(setActionLabels)
  }, [cfg.action_backend, cfg.action_models])

  // Real duration for the first video drives the time-range slider.
  useEffect(() => {
    if (!videos.length) {
      setDuration(0)
      return
    }
    void getVideoInfo(videos[0]).then((r) =>
      setDuration(r.ok ? r.duration : 0),
    )
  }, [videos])

  /** Ranges the user marked in the native Timeline Viewer, via the shared store.
   *  Refreshed on video change and whenever the window regains focus, so ranges
   *  marked in the viewer land here without a manual reload. */
  const refreshAvoidRanges = () => {
    if (!videos.length) {
      setAvoidRanges([])
      return
    }
    void getAvoidRanges(videos[0]).then((r) =>
      setAvoidRanges(r.ok ? r.ranges : []),
    )
  }

  useEffect(() => {
    refreshAvoidRanges()
    window.addEventListener("focus", refreshAvoidRanges)
    return () => window.removeEventListener("focus", refreshAvoidRanges)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [videos])

  // Lifetime analyzed counter (shared stats file with the Qt GUI).
  useEffect(() => {
    void getStats().then((r) => r.ok && setAnalyzed(r.analyzed))
  }, [sessionCount])

  // Preview toggle applies mid-run, matching the Qt checkbox.
  useEffect(() => {
    void setPreview(livePreview)
  }, [livePreview])

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [log])

  const appendLog = (text: string, kind: LogLine["kind"] = "info") =>
    setLog((l) => [...l, { text, kind }])

  const handleEvent = (e: RunEvent) => {
    switch (e.type) {
      case "started":
        appendLog("=== Started ===", "ok")
        break
      case "log":
        appendLog(e.message)
        break
      case "progress":
        if (e.total > 0) setProgress(Math.round((e.current / e.total) * 100))
        setTask(e.detail ? `${e.task} — ${e.detail}` : e.task)
        break
      case "downloaded":
        if (dlRef.current.autoAdd && e.paths.length) {
          setVideos((v) => [...new Set([...v, ...e.paths])])
          appendLog(`➕ Added ${e.paths.length} downloaded video(s)`, "ok")
        }
        break
      case "faces_scanned":
        appendLog(`👤 Found ${e.count} identities`, "ok")
        setFaceRefresh((n) => n + 1)
        break
      case "preview":
        setFrames((f) => {
          const next = [...f, { jpeg: e.jpeg, boxes: e.boxes, sec: e.sec }]
          return next.length > MAX_FRAMES ? next.slice(-MAX_FRAMES) : next
        })
        break
      case "vision_hit":
        appendLog(`🔎 match at ${e.timestamp.toFixed(1)}s`, "ok")
        break
      case "vision_results":
        setVisionResults(e.results)
        break
      case "finished":
        appendLog(`✔ Finished: ${e.output || "(no output)"}`, "ok")
        setSessionCount((n) => n + 1)
        toast.success("Done")
        break
      case "cancelled":
        appendLog("⏹ Cancelled", "err")
        toast("Cancelled")
        break
      case "error":
        appendLog(`✖ ${e.message}`, "err")
        toast.error("Error — see log")
        break
      case "done":
        setRunning(false)
        setTask("")
        wsRef.current?.close()
        wsRef.current = null
        break
    }
  }

  /** Open the events socket and give it a tick to connect before work starts. */
  const beginRun = async () => {
    setLog([])
    setFrames([])
    setProgress(0)
    setRunning(true)
    wsRef.current = openEventSocket(handleEvent)
    await new Promise((r) => setTimeout(r, 150))
  }

  const failRun = (msg: string) => {
    appendLog(`✖ ${msg}`, "err")
    toast.error(msg)
    setRunning(false)
    wsRef.current?.close()
  }

  const onRun = async () => {
    if (!videos.length) return toast.error("Add at least one video")
    if (totalPoints(cfg) === 0 && !cfg.skip_highlights)
      return toast.error("Set at least one scoring point")

    await beginRun()
    const res = await startRun(
      videos,
      toGuiConfig(cfg, output, videos, {
        avoidIds,
        avoidRanges,
        timeRange,
        duration,
      }),
    )
    if (!res.ok) failRun(res.error ?? "Failed to start")
  }

  /** Run -> Pause -> Resume, matching the Qt toggle_run tri-state. */
  const onToggleRun = async () => {
    if (!running) return onRun()
    if (paused) {
      await resumeRun()
      setPaused(false)
      appendLog("▶ Resumed")
    } else {
      await pauseRun()
      setPaused(true)
      appendLog("⏸ Pipeline paused")
    }
  }

  /** urls set = download exactly those (from the picker); otherwise scrape. */
  const onDownload = async (urls?: string[]) => {
    await beginRun()
    const res = await startDownload({
      url: dl.url,
      save_dir: dl.saveDir,
      download_full: dl.downloadFull,
      time_range_start: dl.rangeStart,
      time_range_end: dl.rangeEnd,
      concurrent: dl.concurrent,
      ...(urls?.length ? { video_urls: urls } : {}),
    })
    if (!res.ok) failRun(res.error ?? "Failed to start download")
  }

  const onCancel = async () => {
    await cancelRun()
    appendLog("⏹ Cancellation requested…", "err")
  }

  const addVideos = async () => {
    const picked = await pickVideos()
    if (picked.length) setVideos((v) => [...new Set([...v, ...picked])])
  }

  const launchEditor = async () => {
    if (!videos.length) return toast.error("Add a video first")
    // The viewer is a separate Qt process and takes ~10s to appear, so say so —
    // otherwise the click looks like it did nothing.
    toast("Opening Timeline Viewer — it takes a few seconds to appear…")
    appendLog(`📊 Opening Timeline Viewer for ${basename(videos[0])}…`)
    const res = await openEditor(videos[0])
    if (!res.ok) {
      toast.error(res.error ?? "Could not open the Timeline Viewer")
      appendLog(`✖ ${res.error}`, "err")
    }
  }

  return (
    <div className="mx-auto flex min-h-screen w-full max-w-5xl flex-col gap-5 p-6">
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
        <div className="flex items-center gap-2">
          {analyzed !== null && (
            <span
              className="flex items-center gap-1.5 text-xs text-muted-foreground"
              title="Videos successfully analyzed. Lifetime total persists across sessions."
            >
              <TrendingUp className="size-3.5" />
              <span className="tabular-nums">
                {analyzed} analyzed
                {sessionCount > 0 && ` (session: ${sessionCount})`}
              </span>
            </span>
          )}
          <Button
            variant="outline"
            size="sm"
            onClick={launchEditor}
            disabled={!videos.length}
            title={
              videos.length
                ? "Open the native Timeline Viewer for the first video"
                : "Add a video first"
            }
          >
            <MonitorPlay className="size-4" /> Timeline Viewer
          </Button>
          <Button variant="ghost" size="icon" onClick={toggle} title="Toggle theme">
            {theme === "dark" ? <Sun className="size-4" /> : <Moon className="size-4" />}
          </Button>
        </div>
      </header>

      {/* Input videos */}
      <Card>
        <CardHeader className="flex-row items-center justify-between space-y-0">
          <CardTitle className="text-sm font-medium">Input Videos</CardTitle>
          <div className="flex gap-2">
            {/* Inputs lock during a run, same as the Qt GUI. */}
            <Button size="sm" variant="secondary" onClick={addVideos} disabled={running}>
              <Plus className="size-4" /> Add
            </Button>
            <Button
              size="sm"
              variant="ghost"
              onClick={() => setVideos([])}
              disabled={!videos.length || running}
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
                    className="text-muted-foreground hover:text-destructive disabled:opacity-40"
                    disabled={running}
                    onClick={() => setVideos((l) => l.filter((x) => x !== v))}
                  >
                    <Trash2 className="size-3.5" />
                  </button>
                </li>
              ))}
            </ul>
          )}
          <Separator className="my-4" />
          <div className="grid min-w-0 grid-cols-[auto_minmax(0,1fr)] items-center gap-3">
            <Label className="text-sm text-muted-foreground">Output name</Label>
            <Input
              value={output}
              onChange={(e) => setOutput(e.target.value)}
              disabled={running}
              className="h-8 w-full"
            />
          </div>
        </CardContent>
      </Card>

      <TimeRange state={timeRange} onChange={setTimeRange} duration={duration} />

      <div className="flex flex-wrap items-center gap-x-6 gap-y-2">
        <label className="flex items-center gap-2 text-sm">
          <Checkbox
            checked={livePreview}
            onCheckedChange={(v) => setLivePreview(Boolean(v))}
          />
          Live detection preview
        </label>
        <label className="flex items-center gap-2 text-sm">
          <Checkbox
            checked={cfg.force_reprocess}
            onCheckedChange={(v) => set("force_reprocess", Boolean(v))}
          />
          Force reprocess (ignore cache)
        </label>
        {livePreview && !cfg.force_reprocess && (
          <span className="text-xs text-muted-foreground">
            On an already-analysed video, detection is skipped and the preview
            stays blank — tick Force reprocess to see frames.
          </span>
        )}
      </div>

      {livePreview && <DetectionPreview frames={frames} />}

      {/* Tabs */}
      <Tabs defaultValue="basic" className="min-w-0">
        <TabsList>
          <TabsTrigger value="download">Download</TabsTrigger>
          <TabsTrigger value="basic">Basic</TabsTrigger>
          <TabsTrigger value="transcript">Transcript</TabsTrigger>
          <TabsTrigger value="advanced">Advanced</TabsTrigger>
          <TabsTrigger value="llm">LLM Chat</TabsTrigger>
          <TabsTrigger value="search">Visual Search</TabsTrigger>
          <TabsTrigger value="avoid">Avoid</TabsTrigger>
          <TabsTrigger value="about">About</TabsTrigger>
        </TabsList>

        <TabsContent value="download" className="mt-4">
          <DownloadTab
            settings={dl}
            onChange={setDl}
            onDownload={() => onDownload()}
            onDownloadUrls={(urls) => onDownload(urls)}
            running={running}
          />
        </TabsContent>
        <TabsContent value="basic" className="mt-4">
          <BasicTab
            cfg={cfg}
            set={set}
            objectLabels={objectLabels}
            actionLabels={actionLabels}
          />
        </TabsContent>
        <TabsContent value="transcript" className="mt-4">
          <TranscriptTab cfg={cfg} set={set} />
        </TabsContent>
        <TabsContent value="advanced" className="mt-4">
          <AdvancedTab cfg={cfg} set={set} />
        </TabsContent>
        <TabsContent value="llm" className="mt-4">
          <LlmChatTab
            videoPath={videos[0]}
            backend={llmBackend}
            model={llmModel}
            onBackendChange={setLlmBackend}
            onModelChange={setLlmModel}
          />
        </TabsContent>
        <TabsContent value="search" className="mt-4">
          <VisionSearchTab
            videoPath={videos[0]}
            backend={llmBackend || "ollama"}
            model={llmModel || "llava"}
            running={running}
            results={visionResults}
            onStart={async () => {
              setVisionResults([])
              await beginRun()
            }}
          />
        </TabsContent>
        <TabsContent value="avoid" className="mt-4">
          <AvoidTab
            cfg={cfg}
            set={set}
            onAvoidIdsChange={setAvoidIds}
            videoPath={videos[0]}
            running={running}
            refreshKey={faceRefresh}
            avoidRanges={avoidRanges}
            onAvoidRangesChange={refreshAvoidRanges}
          />
        </TabsContent>
        <TabsContent value="about" className="mt-4">
          <AboutTab />
        </TabsContent>
      </Tabs>

      {/* Run bar — Run / Pause / Resume tri-state, with Cancel alongside. */}
      <div className="flex items-center gap-4">
        <Button
          onClick={onToggleRun}
          disabled={online === false}
          className={
            running
              ? paused
                ? "gap-2 bg-primary text-primary-foreground hover:opacity-90"
                : "gap-2 bg-[#ff8c00] text-white hover:opacity-90"
              : "gap-2 bg-[color:var(--success)] text-white hover:opacity-90"
          }
        >
          {!running ? (
            <>
              <Sparkles className="size-4" /> Run Highlighter
            </>
          ) : paused ? (
            <>
              <Play className="size-4" /> Resume
            </>
          ) : (
            <>
              <Pause className="size-4" /> Pause
            </>
          )}
        </Button>
        {running && (
          <Button variant="destructive" onClick={onCancel} className="gap-2">
            <Square className="size-4" /> Cancel
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
