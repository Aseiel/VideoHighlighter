import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Checkbox } from "@/components/ui/checkbox"
import { NumberField } from "@/components/NumberField"
import { SelectField } from "@/components/SelectField"
import {
  ACTION_BACKENDS,
  R3D_MODELS,
  YOLO_SIZES,
  YOLO_TYPES,
  type HighlighterConfig,
} from "@/lib/config"

interface Props {
  cfg: HighlighterConfig
  set: <K extends keyof HighlighterConfig>(k: K, v: HighlighterConfig[K]) => void
}

export function AdvancedTab({ cfg, set }: Props) {
  return (
    <div className="grid min-w-0 gap-5 md:grid-cols-2 [&>*]:min-w-0">
      <Card>
        <CardHeader>
          <CardTitle className="text-sm font-medium">Motion Recognition</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2.5">
          <NumberField
            label="Frame skip"
            hint="(higher = faster)"
            value={cfg.frame_skip}
            min={1}
            onChange={(v) => set("frame_skip", v)}
          />
          <label className="flex items-center gap-2 text-sm">
            <Checkbox
              checked={cfg.vr_mode}
              onCheckedChange={(v) => set("vr_mode", Boolean(v))}
            />
            VR side-by-side optimization
          </label>
          <p className="text-xs text-muted-foreground">
            Analyses the left half only, for side-by-side VR/3D footage.
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-sm font-medium">Object Recognition</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2.5">
          <NumberField
            label="Frame skip"
            value={cfg.object_frame_skip}
            min={1}
            onChange={(v) => set("object_frame_skip", v)}
          />
          <SelectField
            label="Detector type"
            value={cfg.yolo_type}
            options={YOLO_TYPES}
            onChange={(v) => set("yolo_type", v)}
          />
          <SelectField
            label="Model size"
            value={cfg.yolo_model_size}
            options={YOLO_SIZES}
            onChange={(v) => set("yolo_model_size", v)}
            disabled={cfg.yolo_type === "custom"}
          />
          <NumberField
            label="Confidence threshold"
            hint="(%)"
            value={cfg.obj_confidence}
            min={5}
            onChange={(v) => set("obj_confidence", v)}
          />
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-sm font-medium">Action Recognition</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2.5">
          <NumberField
            label="Frame skip"
            value={cfg.sample_rate}
            min={1}
            onChange={(v) => set("sample_rate", v)}
          />
          <SelectField
            label="Backend"
            value={cfg.action_backend}
            options={ACTION_BACKENDS}
            onChange={(v) => set("action_backend", v)}
          />
          <SelectField
            label="R3D model variant"
            value={cfg.r3d_model}
            options={R3D_MODELS}
            onChange={(v) => set("r3d_model", v)}
            disabled={cfg.action_backend === "openvino"}
          />
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-sm font-medium">Visualization</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <label className="flex items-center gap-2 text-sm">
            <Checkbox
              checked={cfg.draw_object_boxes}
              onCheckedChange={(v) => set("draw_object_boxes", Boolean(v))}
            />
            Draw object bounding boxes
          </label>
          <label className="flex items-center gap-2 text-sm">
            <Checkbox
              checked={cfg.draw_action_labels}
              onCheckedChange={(v) => set("draw_action_labels", Boolean(v))}
            />
            Draw action labels
          </label>
          <p className="text-xs text-muted-foreground">
            Overlays are burned into the temp clips, useful for debugging what
            the detector saw.
          </p>
        </CardContent>
      </Card>
    </div>
  )
}
