import { useEffect, useState } from "react"
import { RefreshCw, UserX, ExternalLink } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Checkbox } from "@/components/ui/checkbox"
import { Badge } from "@/components/ui/badge"
import { SelectField } from "@/components/SelectField"
import { AVOID_METHODS, type HighlighterConfig } from "@/lib/config"
import { getFaces, setFaceAvoid, openEditor, type FaceIdentity } from "@/lib/api"
import { toast } from "sonner"

interface Props {
  cfg: HighlighterConfig
  set: <K extends keyof HighlighterConfig>(k: K, v: HighlighterConfig[K]) => void
  onAvoidIdsChange: (ids: string[]) => void
}

export function AvoidTab({ cfg, set, onAvoidIdsChange }: Props) {
  const [faces, setFaces] = useState<FaceIdentity[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

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
  }, [])

  const toggle = async (f: FaceIdentity, next: boolean) => {
    // Optimistic: reflect immediately, revert if the bank write fails.
    setFaces((cur) =>
      cur.map((x) => (x.id === f.id ? { ...x, avoid: next } : x)),
    )
    const res = await setFaceAvoid(f.id, next)
    if (!res.ok) {
      toast.error(res.error ?? "Could not update face bank")
      setFaces((cur) =>
        cur.map((x) => (x.id === f.id ? { ...x, avoid: !next } : x)),
      )
      return
    }
    const ids = faces
      .map((x) => (x.id === f.id ? { ...x, avoid: next } : x))
      .filter((x) => x.avoid)
      .map((x) => x.id)
    onAvoidIdsChange(ids)
  }

  const avoidCount = faces.filter((f) => f.avoid).length

  return (
    <Card>
      <CardHeader className="flex-row items-center justify-between space-y-0">
        <CardTitle className="flex items-center gap-2 text-sm font-medium">
          <UserX className="size-4" /> Avoid People
          {avoidCount > 0 && <Badge>{avoidCount} avoided</Badge>}
        </CardTitle>
        <Button size="sm" variant="secondary" onClick={refresh} disabled={loading}>
          <RefreshCw className={loading ? "size-4 animate-spin" : "size-4"} />
          Refresh
        </Button>
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
          People you name in the Timeline Viewer (right-click a face → Name) show
          up here. Tick someone to exclude them from generated highlights.
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
                No faces in the bank yet.
              </p>
              <Button
                size="sm"
                variant="outline"
                onClick={async () => {
                  const res = await openEditor()
                  if (!res.ok) toast.error(res.error ?? "Could not open editor")
                }}
              >
                <ExternalLink className="size-4" /> Open Timeline Viewer to scan
              </Button>
            </div>
          ) : (
            <ul className="divide-y">
              {faces.map((f) => (
                <li
                  key={f.id}
                  className="flex items-center justify-between px-3 py-2 text-sm"
                >
                  <label className="flex min-w-0 items-center gap-2">
                    <Checkbox
                      checked={f.avoid}
                      onCheckedChange={(v) => toggle(f, Boolean(v))}
                      disabled={!cfg.avoid_enabled}
                    />
                    <span className="truncate">{f.label}</span>
                  </label>
                  <span className="shrink-0 text-xs text-muted-foreground">
                    seen {f.count}×
                  </span>
                </li>
              ))}
            </ul>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
