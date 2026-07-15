import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Checkbox } from "@/components/ui/checkbox"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { NumberField } from "@/components/NumberField"
import { totalPoints, type HighlighterConfig } from "@/lib/config"

interface Props {
  cfg: HighlighterConfig
  set: <K extends keyof HighlighterConfig>(k: K, v: HighlighterConfig[K]) => void
  objectLabels: string[]
  actionLabels: string[]
}

export function BasicTab({ cfg, set, objectLabels, actionLabels }: Props) {
  return (
    <div className="space-y-5">
      <div className="grid min-w-0 gap-5 md:grid-cols-2 [&>*]:min-w-0">
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
            <p className="pt-1 text-xs text-muted-foreground">
              Keyword and transcript points live in the Transcript tab — they
              only count while transcript is enabled.
            </p>
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
            <p className="text-xs text-muted-foreground">
              {cfg.clip_time === 0
                ? "Auto mode: clip boundaries come from signal structure (actions, scene cuts, peaks)."
                : `Fixed mode: every clip is ${cfg.clip_time}s long.`}
            </p>
            <NumberField label="Auto min clip" hint="(s)" value={cfg.auto_min_clip} step={0.5} onChange={(v) => set("auto_min_clip", v)} />
            <NumberField label="Auto max clip" hint="(s)" value={cfg.auto_max_clip} step={0.5} onChange={(v) => set("auto_max_clip", v)} />
            <NumberField label="Merge gap" hint="(s)" value={cfg.auto_merge_gap} step={0.5} onChange={(v) => set("auto_merge_gap", v)} />
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-sm font-medium">Detection Targets</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="grid min-w-0 grid-cols-[4.5rem_minmax(0,1fr)] items-center gap-3">
            <Label className="text-sm text-muted-foreground">Objects</Label>
            <Input
              list="object-labels"
              value={cfg.highlight_objects}
              onChange={(e) => set("highlight_objects", e.target.value)}
              placeholder="person, sports ball, dog"
              className="h-8 w-full"
            />
            <datalist id="object-labels">
              {objectLabels.map((l) => (
                <option key={l} value={l} />
              ))}
            </datalist>
          </div>
          <div className="grid min-w-0 grid-cols-[4.5rem_minmax(0,1fr)] items-center gap-3">
            <Label className="text-sm text-muted-foreground">Actions</Label>
            <Input
              list="action-labels"
              value={cfg.interesting_actions}
              onChange={(e) => set("interesting_actions", e.target.value)}
              placeholder="high jump, high kick, archery"
              className="h-8 w-full"
            />
            <datalist id="action-labels">
              {actionLabels.map((l) => (
                <option key={l} value={l} />
              ))}
            </datalist>
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
            {/* Force reprocess lives on the main screen next to Live preview,
                where the Qt app puts it and where it's actually needed. */}
            <label className="flex items-center gap-2 text-sm">
              <Checkbox
                checked={cfg.skip_highlights}
                onCheckedChange={(v) => set("skip_highlights", Boolean(v))}
              />
              Skip highlights
            </label>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
