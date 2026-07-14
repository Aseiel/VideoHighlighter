import { useEffect, useState } from "react"
import { RefreshCw, UserX, ExternalLink, ScanFace, Trash2, X } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Checkbox } from "@/components/ui/checkbox"
import { Badge } from "@/components/ui/badge"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { SelectField } from "@/components/SelectField"
import { AVOID_METHODS, type HighlighterConfig } from "@/lib/config"
import {
  getFaces,
  setFaceAvoid,
  removeFace,
  nameFace,
  clearFaces,
  scanFaces,
  openEditor,
  type FaceIdentity,
} from "@/lib/api"
import { toast } from "sonner"

interface Props {
  cfg: HighlighterConfig
  set: <K extends keyof HighlighterConfig>(k: K, v: HighlighterConfig[K]) => void
  onAvoidIdsChange: (ids: string[]) => void
  /** First input video — the scan target, matching the Qt single-video rule. */
  videoPath?: string
  running: boolean
  /** Bumped by App when a faces_scanned event arrives, to trigger a refresh. */
  refreshKey?: number
}

export function AvoidTab({
  cfg,
  set,
  onAvoidIdsChange,
  videoPath,
  running,
  refreshKey,
}: Props) {
  const [faces, setFaces] = useState<FaceIdentity[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [clearOpen, setClearOpen] = useState(false)

  const refresh = async () => {
    setLoading(true)
    const res = await getFaces()
    setLoading(false)
    if (!res.ok) {
      setError(res.error ?? "Face bank unavailable")
      setFaces([])
      return
    }
    setError(null)
    setFaces(res.identities)
    onAvoidIdsChange(res.identities.filter((f) => f.avoid).map((f) => f.id))
  }

  useEffect(() => {
    void refresh()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [refreshKey])

  const syncAvoidIds = (next: FaceIdentity[]) =>
    onAvoidIdsChange(next.filter((x) => x.avoid).map((x) => x.id))

  const toggle = async (f: FaceIdentity, next: boolean) => {
    // Optimistic: reflect immediately, revert if the bank write fails.
    const optimistic = faces.map((x) =>
      x.id === f.id ? { ...x, avoid: next } : x,
    )
    setFaces(optimistic)
    syncAvoidIds(optimistic)
    const res = await setFaceAvoid(f.id, next)
    if (!res.ok) {
      toast.error(res.error ?? "Could not update face bank")
      void refresh()
    }
  }

  const rename = async (f: FaceIdentity) => {
    const name = window.prompt("Name this person:", f.name)
    if (name === null) return
    const res = await nameFace(f.id, name)
    if (!res.ok) return toast.error(res.error ?? "Could not set name")
    if (res.merged_into) toast.success("Merged into the existing person")
    void refresh()
  }

  const remove = async (f: FaceIdentity) => {
    const res = await removeFace(f.id)
    if (!res.ok) return toast.error("Could not remove")
    void refresh()
  }

  const doClear = async (keepNamed: boolean) => {
    setClearOpen(false)
    const res = await clearFaces(keepNamed)
    if (!res.ok) return toast.error(res.error ?? "Could not clear")
    toast.success(`Cleared — ${res.remaining} kept`)
    void refresh()
  }

  const scan = async () => {
    if (!videoPath) return toast.error("Add a video first")
    const res = await scanFaces(videoPath)
    if (!res.ok) toast.error(res.error ?? "Could not start scan")
    else toast("Scanning for faces — see the log below")
  }

  const avoidCount = faces.filter((f) => f.avoid).length
  const namedCount = faces.filter((f) => f.name).length

  return (
    <Card>
      <CardHeader className="flex-row items-center justify-between space-y-0">
        <CardTitle className="flex items-center gap-2 text-sm font-medium">
          <UserX className="size-4" /> Avoid People
          {avoidCount > 0 && <Badge>{avoidCount} avoided</Badge>}
        </CardTitle>
        <div className="flex gap-2">
          <Button size="sm" variant="secondary" onClick={refresh} disabled={loading}>
            <RefreshCw className={loading ? "size-4 animate-spin" : "size-4"} />
            Refresh
          </Button>
          <Button size="sm" variant="secondary" onClick={scan} disabled={running}>
            <ScanFace className="size-4" /> Scan video
          </Button>
          <Button
            size="sm"
            variant="ghost"
            onClick={() => setClearOpen(true)}
            disabled={!faces.length}
          >
            <Trash2 className="size-4" /> Clear
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <label className="flex items-center gap-2 text-sm">
          <Checkbox
            checked={cfg.avoid_enabled}
            onCheckedChange={(v) => set("avoid_enabled", Boolean(v))}
          />
          Enable face recognition
        </label>
        <p className="text-xs text-muted-foreground">
          Scan a video to collect everyone who appears, then tick who to exclude.
          You can also name people here or in the Timeline Viewer.
        </p>

        <SelectField
          label="When found"
          value={cfg.avoid_method}
          options={AVOID_METHODS}
          onChange={(v) => set("avoid_method", v)}
          disabled={!cfg.avoid_enabled}
        />

        <div className="rounded-md border">
          {error ? (
            <p className="p-4 text-center text-sm text-destructive">{error}</p>
          ) : faces.length === 0 ? (
            <div className="space-y-3 p-6 text-center">
              <p className="text-sm text-muted-foreground">
                No faces in the bank yet. Scan a video, or name faces in the
                Timeline Viewer.
              </p>
              <Button
                size="sm"
                variant="outline"
                onClick={async () => {
                  const res = await openEditor(videoPath)
                  if (!res.ok) toast.error(res.error ?? "Could not open editor")
                }}
              >
                <ExternalLink className="size-4" /> Open Timeline Viewer
              </Button>
            </div>
          ) : (
            <ul className="divide-y">
              {faces.map((f) => (
                <li key={f.id} className="flex items-center gap-3 px-3 py-2 text-sm">
                  <Checkbox
                    checked={f.avoid}
                    onCheckedChange={(v) => toggle(f, Boolean(v))}
                    disabled={!cfg.avoid_enabled}
                  />
                  {f.thumb ? (
                    <img
                      src={`data:image/jpeg;base64,${f.thumb}`}
                      alt=""
                      className="size-10 shrink-0 rounded object-cover"
                    />
                  ) : (
                    <div className="size-10 shrink-0 rounded bg-muted" />
                  )}
                  <button
                    className="min-w-0 flex-1 truncate text-left hover:underline"
                    onClick={() => rename(f)}
                    title="Click to name"
                  >
                    <span className="font-medium">{f.label}</span>
                  </button>
                  <span className="shrink-0 text-xs text-muted-foreground">
                    seen {f.count}×
                  </span>
                  <button
                    className="shrink-0 text-muted-foreground hover:text-destructive"
                    onClick={() => remove(f)}
                    title="Remove"
                  >
                    <X className="size-4" />
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
        {faces.length > 0 && (
          <p className="text-xs text-muted-foreground">
            {faces.length} people · {namedCount} named · {avoidCount} avoided
          </p>
        )}
      </CardContent>

      <Dialog open={clearOpen} onOpenChange={setClearOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Clear the face bank ({faces.length} identities)?</DialogTitle>
            <DialogDescription>Choose what to remove.</DialogDescription>
          </DialogHeader>
          <DialogFooter className="gap-2">
            <Button variant="ghost" onClick={() => setClearOpen(false)}>
              Cancel
            </Button>
            <Button variant="secondary" onClick={() => doClear(true)}>
              Keep named / avoided
            </Button>
            <Button variant="destructive" onClick={() => doClear(false)}>
              Clear everything
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </Card>
  )
}
