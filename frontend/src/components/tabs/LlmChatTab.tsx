import { useEffect, useRef, useState } from "react"
import { Bot, RefreshCw, Send, User } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"
import { SelectField } from "@/components/SelectField"
import { getLlmBackends, llmChat } from "@/lib/api"
import { basename } from "@/lib/files"

interface Props {
  /** First selected video — its cached analysis becomes the chat context. */
  videoPath?: string
  /** Lifted so Visual Search uses the same backend/model selection. */
  backend: string
  model: string
  onBackendChange: (v: string) => void
  onModelChange: (v: string) => void
}

type Msg = { role: "user" | "assistant"; text: string }

export function LlmChatTab({
  videoPath,
  backend,
  model,
  onBackendChange,
  onModelChange,
}: Props) {
  const [backends, setBackends] = useState<string[]>([])
  const [models, setModels] = useState<string[]>([])
  const setBackend = onBackendChange
  const setModel = onModelChange
  const [msgs, setMsgs] = useState<Msg[]>([])
  const [input, setInput] = useState("")
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const endRef = useRef<HTMLDivElement | null>(null)

  const refresh = async () => {
    const res = await getLlmBackends()
    if (!res.ok) {
      setError(res.error ?? "LLM stack unavailable")
      return
    }
    setError(null)
    setBackends(res.backends)
    setModels(res.ollama_models)
    if (res.backends.length && !backend) setBackend(res.backends[0])
    if (res.ollama_models.length && !model) setModel(res.ollama_models[0])
  }

  useEffect(() => {
    void refresh()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [msgs])

  const send = async () => {
    const text = input.trim()
    if (!text || !backend || !model) return
    setInput("")
    setMsgs((m) => [...m, { role: "user", text }])
    setBusy(true)
    const res = await llmChat(backend, model, text, videoPath)
    setBusy(false)
    setMsgs((m) => [
      ...m,
      {
        role: "assistant",
        text: res.ok ? (res.answer ?? "") : `Error: ${res.error}`,
      },
    ])
  }

  const ready = backends.length > 0 && Boolean(model)

  return (
    <Card className="flex min-h-0 flex-col">
      <CardHeader className="space-y-3">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2 text-sm font-medium">
            <Bot className="size-4" /> LLM Chat
            {videoPath && (
              <Badge variant="secondary" className="font-normal">
                context: {basename(videoPath)}
              </Badge>
            )}
          </CardTitle>
          <Button size="sm" variant="secondary" onClick={refresh}>
            <RefreshCw className="size-4" /> Refresh
          </Button>
        </div>
        {error ? (
          <p className="text-xs text-destructive">{error}</p>
        ) : backends.length === 0 ? (
          <p className="text-xs text-muted-foreground">
            No local LLM backend found. Install Ollama (or llama-cpp-python) and
            hit Refresh.
          </p>
        ) : (
          <div className="grid gap-2">
            <SelectField
              label="Backend"
              value={backend}
              options={backends}
              onChange={setBackend}
            />
            {models.length > 0 ? (
              <SelectField
                label="Model"
                value={model}
                options={models}
                onChange={setModel}
              />
            ) : (
              <div className="grid min-w-0 grid-cols-[minmax(0,1fr)_14rem] items-center gap-3">
                <span className="text-sm text-muted-foreground">Model</span>
                <Input
                  value={model}
                  onChange={(e) => setModel(e.target.value)}
                  placeholder="llama3.2"
                  className="h-8"
                />
              </div>
            )}
          </div>
        )}
      </CardHeader>
      <CardContent className="flex min-h-0 flex-1 flex-col gap-3">
        <ScrollArea className="h-72 rounded-md border p-3">
          {msgs.length === 0 ? (
            <p className="py-8 text-center text-sm text-muted-foreground">
              Ask about the selected video. Answers use its cached analysis.
            </p>
          ) : (
            <div className="space-y-3">
              {msgs.map((m, i) => (
                <div key={i} className="flex gap-2 text-sm">
                  <div className="mt-0.5 shrink-0 text-muted-foreground">
                    {m.role === "user" ? (
                      <User className="size-4" />
                    ) : (
                      <Bot className="size-4" />
                    )}
                  </div>
                  <p className="min-w-0 whitespace-pre-wrap break-words">
                    {m.text}
                  </p>
                </div>
              ))}
              {busy && (
                <p className="text-sm text-muted-foreground">Thinking…</p>
              )}
              <div ref={endRef} />
            </div>
          )}
        </ScrollArea>
        <div className="flex gap-2">
          <Input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && !busy && send()}
            placeholder={
              ready ? "Ask about this video…" : "Select a backend and model first"
            }
            disabled={!ready || busy}
          />
          <Button onClick={send} disabled={!ready || busy || !input.trim()}>
            <Send className="size-4" />
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}
