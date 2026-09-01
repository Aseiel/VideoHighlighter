import { useEffect, useState } from "react"
import { Plus, Save, X } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { getCompositionRules, saveCompositionRules, type CompRule } from "@/lib/api"
import { toast } from "sonner"

const BLANK: CompRule = {
  name: "",
  label: "",
  source: "",
  region: "",
  min_count: 1,
  max_count: 999,
  window_secs: 0.75,
  persist_secs: 0.5,
}

export function CompositionRules() {
  const [rules, setRules] = useState<CompRule[]>([])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    void getCompositionRules().then((r) => r.ok && setRules(r.rules))
  }, [])

  const upd = (i: number, patch: Partial<CompRule>) =>
    setRules((rs) => rs.map((r, j) => (j === i ? { ...r, ...patch } : r)))

  const save = async () => {
    setLoading(true)
    const res = await saveCompositionRules(rules)
    setLoading(false)
    if (res.ok) toast.success(`Saved ${res.events} event(s) to composition_rules.yaml`)
    else toast.error(res.error ?? "Could not save rules")
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-sm font-medium">Composition Rules</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <p className="text-xs text-muted-foreground">
          Compose higher-level actions from spatial relationships between detected
          objects — e.g. if object A appears inside region B enough times, fire
          action X. Rows sharing an Event Name must all be satisfied together (AND).
          Window smooths over flicker; Persist keeps an object alive through
          occlusion. Saved to composition_rules.yaml.
        </p>

        <div className="overflow-x-auto rounded-md border">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="min-w-28">Event Name</TableHead>
                <TableHead className="min-w-28">Display Label</TableHead>
                <TableHead className="min-w-24">Source</TableHead>
                <TableHead className="min-w-24">Region</TableHead>
                <TableHead className="w-20">Min</TableHead>
                <TableHead className="w-20">Max</TableHead>
                <TableHead className="w-24">Window (s)</TableHead>
                <TableHead className="w-24">Persist (s)</TableHead>
                <TableHead className="w-10" />
              </TableRow>
            </TableHeader>
            <TableBody>
              {rules.length === 0 ? (
                <TableRow>
                  <TableCell
                    colSpan={9}
                    className="text-center text-sm text-muted-foreground"
                  >
                    No rules yet
                  </TableCell>
                </TableRow>
              ) : (
                rules.map((r, i) => (
                  <TableRow key={i}>
                    {(["name", "label", "source", "region"] as const).map((f) => (
                      <TableCell key={f} className="p-1">
                        <Input
                          value={r[f]}
                          onChange={(e) => upd(i, { [f]: e.target.value })}
                          className="h-7"
                        />
                      </TableCell>
                    ))}
                    {(
                      [
                        ["min_count", 1],
                        ["max_count", 1],
                        ["window_secs", 0.25],
                        ["persist_secs", 0.25],
                      ] as const
                    ).map(([f, step]) => (
                      <TableCell key={f} className="p-1">
                        <Input
                          type="number"
                          step={step}
                          value={r[f]}
                          onChange={(e) => upd(i, { [f]: Number(e.target.value) })}
                          className="h-7 text-right tabular-nums"
                        />
                      </TableCell>
                    ))}
                    <TableCell className="p-1">
                      <button
                        className="text-destructive hover:opacity-70"
                        onClick={() =>
                          setRules((rs) => rs.filter((_, j) => j !== i))
                        }
                        title="Remove rule"
                      >
                        <X className="size-4" />
                      </button>
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </div>

        <div className="flex justify-between">
          <Button
            size="sm"
            variant="secondary"
            onClick={() => setRules((rs) => [...rs, { ...BLANK }])}
          >
            <Plus className="size-4" /> Add Rule
          </Button>
          <Button size="sm" onClick={save} disabled={loading}>
            <Save className="size-4" /> Save Rules
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}
