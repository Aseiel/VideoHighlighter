import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"

interface Props {
  label: string
  value: number
  onChange: (v: number) => void
  min?: number
  step?: number
  hint?: string
}

export function NumberField({ label, value, onChange, min = 0, step = 1, hint }: Props) {
  return (
    // Label and field sit together rather than at opposite ends of the card.
    // Stretching a 2-digit number box across a 450px row leaves a gulf the eye
    // has to jump, and rows of that are what make a settings list read as filler.
    // min-w-0 + truncate: a long label must shorten, not widen the row past the
    // card (flex/grid children default to min-width:auto).
    <div className="flex min-w-0 items-center gap-3">
      <Label className="min-w-0 flex-1 truncate text-sm font-normal text-muted-foreground">
        {label}
        {hint && <span className="ml-1 text-xs opacity-60">{hint}</span>}
      </Label>
      <Input
        type="number"
        min={min}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="h-7 w-20 shrink-0 text-right tabular-nums"
      />
    </div>
  )
}
