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
  MonitorPlay,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
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
import { DEFAULT_CONFIG, toGuiConfig, totalPoints } from "@/lib/config"
import type { HighlighterConfig } from "@/lib/config"
import {
  startRun,
  startDownload,
  cancelRun,
  openEventSocket,
  getHealth,
  getLabels,
  openEditor,
  type RunEvent,
} from "@/lib/api"
import { BasicTab } from "@/components/tabs/BasicTab"
import { TranscriptTab } from "@/components/tabs/TranscriptTab"
import { AdvancedTab } from "@/components/tabs/AdvancedTab"
import { AvoidTab } from "@/components/tabs/AvoidTab"
import { LlmChatTab } from "@/components/tabs/LlmChatTab"
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
  const [objectLabels, setObjectLabels] = useState<string[]>([])
  const [actionLabels, setActionLabels] = useState<string[]>([])
  const [running, setRunning] = useState(false)
  const [progress, setProgress] = useState(0)
  const [task, setTask] = useState("")
  const [log, setLog] = useState<LogLine[]>([])
  const [online, setOnline] = useState<boolean | null>(null)
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
        })
        .catch(() => alive && setOnline(false))
    check()
    const id = setInterval(check, 4000)
    return () => {
      alive = false
      clearInterval(id)
    }
  }, [])

  // Label vocabularies for the detection autocomplete.
  useEffect(() => {
    void getLabels("objects").then(setObjectLabels)
    void getLabels("actions").then(setActionLabels)
  }, [])

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
      case "finished":
        appendLog(`✔ Finished: ${e.output || "(no output)"}`, "ok")
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
    const res = await startRun(videos, toGuiConfig(cfg, output, videos, avoidIds))
    if (!res.ok) failRun(res.error ?? "Failed to start")
  }

  const onDownload = async () => {
    await beginRun()
    const res = await startDownload({
      url: dl.url,
      save_dir: dl.saveDir,
      download_full: dl.downloadFull,
      time_range_start: dl.rangeStart,
      time_range_end: dl.rangeEnd,
      concurrent: dl.concurrent,
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
    const res = await openEditor(videos[0])
    if (res.ok) toast.success("Opening Timeline Viewer…")
    else toast.error(res.error ?? "Could not open the editor")
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
          <Button
            variant="outline"
            size="sm"
            onClick={launchEditor}
            title="Open the native timeline viewer / editor"
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
          <div className="grid min-w-0 grid-cols-[auto_minmax(0,1fr)] items-center gap-3">
            <Label className="text-sm text-muted-foreground">Output name</Label>
            <Input
              value={output}
              onChange={(e) => setOutput(e.target.value)}
              className="h-8 w-full"
            />
          </div>
        </CardContent>
      </Card>

      {/* Tabs */}
      <Tabs defaultValue="basic" className="min-w-0">
        <TabsList>
          <TabsTrigger value="download">Download</TabsTrigger>
          <TabsTrigger value="basic">Basic</TabsTrigger>
          <TabsTrigger value="transcript">Transcript</TabsTrigger>
          <TabsTrigger value="advanced">Advanced</TabsTrigger>
          <TabsTrigger value="llm">LLM Chat</TabsTrigger>
          <TabsTrigger value="avoid">Avoid</TabsTrigger>
        </TabsList>

        <TabsContent value="download" className="mt-4">
          <DownloadTab
            settings={dl}
            onChange={setDl}
            onDownload={onDownload}
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
          <LlmChatTab videoPath={videos[0]} />
        </TabsContent>
        <TabsContent value="avoid" className="mt-4">
          <AvoidTab cfg={cfg} set={set} onAvoidIdsChange={setAvoidIds} />
        </TabsContent>
      </Tabs>

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
